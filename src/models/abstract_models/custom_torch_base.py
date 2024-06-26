"""
Filename: custom_torch_base.py

Authors: Hakima Laribi
         Nicolas Raymond

Description: Defines the abstract class TorchCustomModel from which all custom pytorch models
             implemented for the project must inherit. This class allows to store common
             function of all pytorch models.

"""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import tensor, ones
from torch.nn import BatchNorm1d, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.data.processing.datasets import HOMRDataset, LightHOMRDataset
from src.data.processing.sampling import MaskType, BatchSampler
from src.evaluating.early_stopping import EarlyStopper
from src.utils.metric_scores import Metric
from src.utils.visualization import visualize_epoch_progression


class TorchCustomModel(Module, ABC):
    """
    Abstract class used to store common attributes
    and methods of torch models implemented in the project
    """

    def __init__(self,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 output_size: int,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 additional_input_args: Optional[List[Any]] = None,
                 verbose: bool = True):
        """
        Sets the protected attributes and creates an embedding block if required

        Args:
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric of our model (Ex. accuracy, mean absolute error)
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns_to_anonymize in the dataset,
                          cont idx are assumed to be range(num_cont_col)
            cat_idx: idx of categorical columns_to_anonymize in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            additional_input_args: list of arguments that must be also considered when validating
                                   input arguments
            verbose: true if we want to print the evaluating progress
        """
        self._device = torch.device('cpu')
        # We validate input arguments (check if there are continuous or categorical inputs)
        additional_input_args = additional_input_args if additional_input_args is not None else []
        self._validate_input_args([num_cont_col, cat_sizes, *additional_input_args])

        # Call of parent's constructor
        Module.__init__(self)
        # Settings of general protected attributes
        self._alpha = alpha
        self._beta = beta
        self._criterion = criterion
        self._criterion_name = criterion_name
        self._eval_metric = eval_metric
        self._evaluations = {i: {self._criterion_name: [],
                                 self._eval_metric.name: []} for i in [MaskType.TRAIN, MaskType.VALID]}
        self._input_size = num_cont_col if num_cont_col is not None else 0
        self._optimizer = None
        self._output_size = output_size
        self._verbose = verbose

        # Settings of protected attributes related to entity embedding
        self._cat_idx = cat_idx if cat_idx is not None else []
        self._cont_idx = list(range(num_cont_col))
        self._embedding_block = None

        # Initialization of a protected method
        self._update_weights = None


    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def device(self):
        return self._device

    def _create_validation_objects(self,
                                   dataset: HOMRDataset,
                                   valid_batch_size: Optional[int],
                                   patience: int
                                   ) -> Tuple:
        """
        Creates the objects needed for validation during the evaluating process

        Args:
            dataset: HOMRDataset used to feed the dataloader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            patience: number of consecutive epochs without improvement allowed

        Returns: EarlyStopper, (Dataloader, HOMRDataset)

        """
        # We create the valid dataloader (if valid size != 0)
        valid_size, valid_data, early_stopper = len(dataset.valid_mask), None, None

        if valid_size != 0:

            # We check if a valid batch size was provided
            valid_bs = valid_batch_size if valid_batch_size is not None else valid_size

            # We create the valid loader
            valid_bs = min(valid_size, valid_bs)

            # Creation of the light dataset containing only the data and targets
            x, y, idx = dataset[dataset.valid_mask]
            light_dataset = LightHOMRDataset(x, y, idx, dataset.temporal_analysis)

            # Creation of evaluating loader
            if dataset.temporal_analysis:
                sampler = BatchSampler(x, idx, batch_size=valid_bs)
                valid_data = DataLoader(light_dataset,
                                        batch_size=1,
                                        batch_sampler=sampler,
                                        drop_last=False,
                                        shuffle=False)

            else:
                valid_data = DataLoader(light_dataset, batch_size=valid_bs)

            early_stopper = EarlyStopper(patience, self._eval_metric.direction)

        return early_stopper, valid_data

    def _disable_running_stats(self) -> None:
        """
        Disables batch norm momentum when executing SAM optimization step

        Returns: None
        """
        self.apply(self.disable_module_running_stats)

    def _enable_running_stats(self) -> None:
        """
        Restores batch norm momentum when executing SAM optimization step

        Returns: None
        """
        self.apply(self.enable_module_running_stats)

    def _basic_weight_update(self, sample_weights: tensor,
                             x: List[tensor],
                             y: tensor,
                             idx,
                             pos_idx: Optional[List[int]] = None) -> Tuple[tensor, tensor, float]:
        """
        Executes a weights update without using Sharpness-Aware Minimization (SAM)

        Args:
            sample_weights: weights of each sample associated to a batch
            x: list of arguments taken for the forward pass (HeteroGraph and (N', D) tensor with batch inputs)
            y: (N',) ground truth associated to a batch
            pos_idx: dictionary that maps the original dataset's idx to their current
                     position in the mask used for the forward pass (used only with GNNs)

        Returns: (N',) tensor with predictions, evaluating loss
        """

        # We compute the predictions and put them in the appropriate device
        pred = self(*x).squeeze(dim=-1)
        pred = pred if pos_idx is None else pred[pos_idx]
        pred = pred.to(self._device)

        # For temporal analysis, we compute the loss on the last sample of the sequence only
        flatten_mask = []
        relative_indexes = []
        initial_index = 0
        # Get relative indexes of the last samples to get the target and prediction corresponding
        for i, indexes in enumerate(idx):
            flatten_mask.append(indexes[-1])
            relative_indexes.append(initial_index + len(indexes) - 1)
            initial_index += len(indexes)

        if len(pred.size()) == 0:
            pred = pred.unsqueeze(0)

        y = y[relative_indexes]
        pred = pred[relative_indexes]

        # We execute a single forward-backward pass
        loss = self.loss(sample_weights, pred, y)
        loss.backward()
        self._optimizer.step()

        return y, pred, loss.item()

    def _generate_progress_func(self, max_epochs: int) -> Callable:
        """
        Builds a function that updates the evaluating progress in the terminal

        Args:
            max_epochs: maximum number of evaluating epochs

        Returns: function
        """
        if self._verbose:
            def update_progress(epoch: int,
                                train_epoch_loss: float,
                                valid_epoch_loss: float,
                                train_epoch_eval: float,
                                valid_epoch_eval: float):
                if (epoch + 1) % 5 == 0 or (epoch + 1) == max_epochs:
                    print(f"Epoch {epoch + 1} - Train Loss : {round(train_epoch_loss, 4)} ========= "
                          f"Train {self._eval_metric.name} : {round(train_epoch_eval, 4)} ========= "
                          f"Valid Loss : {round(valid_epoch_loss, 4)} ========="
                          f"Valid {self._eval_metric.name} : {round(valid_epoch_eval, 4)}")

        else:
            def update_progress(*args):
                pass

        return update_progress

    def fit(self,
            dataset: Union[HOMRDataset, Tuple[HOMRDataset, tensor, tensor]],
            lr: float,
            batch_size: int = 55,
            valid_batch_size: Optional[int] = None,
            max_epochs: int = 200,
            patience: int = 15,
            sample_weights: Optional[tensor] = None) -> None:
        """
        Fits the model to the evaluating data

        Args:
            dataset: Dataset used to feed the dataloaders
            lr: learning rate
            batch_size: size of the batches in the evaluating loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: Maximum number of epochs for evaluating
            patience: Number of consecutive epochs without improvement allowed
            sample_weights: (N,) tensor with weights of the samples in the dataset

        Returns: None
        """

        # We create the evaluating objects
        train_data = self._create_train_objects(dataset, batch_size)

        # We create the objects needed for validation
        early_stopper, valid_data = self._create_validation_objects(dataset, valid_batch_size, patience)

        # We init the update function
        update_progress = self._generate_progress_func(max_epochs)

        # We set the optimizer
        self._update_weights = self._basic_weight_update
        self._optimizer = Adam(self.parameters(), lr=lr)

        # We execute the epochs
        for epoch in range(max_epochs):

            # We calculate evaluating and validation mean epoch loss on all batches
            mean_epoch_loss, mean_epoch_eval = self._execute_train_step(train_data, sample_weights)
            earlystopping, mean_epoch_loss_validation, mean_epoch_eval_validation = self._execute_valid_step(valid_data,
                                                                                 sample_weights, early_stopper)

            update_progress(epoch, mean_epoch_loss, mean_epoch_loss_validation, mean_epoch_eval, mean_epoch_eval_validation)

            # Apply early stopping if needed
            if earlystopping:
                print(f"\nEarly stopping occurred at epoch {epoch} with best_epoch = {epoch - patience}"
                      f" and best_val_{self._eval_metric.name} = {round(early_stopper.best_val_score, 4)}")
                break

        if early_stopper is not None:
            # We extract best params and remove checkpoint file
            self.load_state_dict(early_stopper.get_best_params())
            early_stopper.remove_checkpoint()

    def loss(self,
             sample_weights: tensor,
             pred: tensor,
             y: tensor) -> tensor:
        """
        Calls the criterion and add the elastic penalty

        Args:
            sample_weights: (N,) tensor with weights of samples on which we calculate the loss
            pred: (N, C) tensor if classification with C classes, (N,) tensor for regression
            y: (N,) tensor with targets

        Returns: tensor with loss value
        """
        # Computations of penalties
        l1_penalty, l2_penalty = tensor(0.), tensor(0.)
        for _, w in self.named_parameters():
            l1_penalty = l1_penalty + w.abs().sum()
            l2_penalty = l2_penalty + w.pow(2).sum()

        # Computation of loss without reduction
        loss = self._criterion(pred, y.float())  # (N,) tensor

        # Store variables in the appropriate device
        loss = loss.to(self._device)

        l1_penalty = l1_penalty.to(self._device)
        l2_penalty = l2_penalty.to(self._device)
        if isinstance(self._alpha, float):
            self._alpha = tensor(self._alpha).to(self._device)
            self._beta = tensor(self._beta).to(self._device)

        if sample_weights is None:
            sample_weights = self.samples_weights(y)

        sample_weights = sample_weights.to(self._device)

        # Computation of loss reduction + elastic penalty
        m = (loss * sample_weights).sum() + self._alpha * l1_penalty + self._beta * l2_penalty
        return m

    def plot_evaluations(self, save_path: Optional[str] = None) -> None:
        """
        Plots the evaluating and valid curves saved

        Args:
            save_path: path were the figures will be saved

        Returns: None
        """
        # Extraction of data
        train_loss = self._evaluations[MaskType.TRAIN][self._criterion_name]
        train_metric = self._evaluations[MaskType.TRAIN][self._eval_metric.name]
        valid_loss = self._evaluations[MaskType.VALID][self._criterion_name]
        valid_metric = self._evaluations[MaskType.VALID][self._eval_metric.name]

        # Figure construction
        visualize_epoch_progression(train_history=[train_loss, train_metric],
                                    valid_history=[valid_loss, valid_metric],
                                    progression_type=[self._criterion_name, self._eval_metric.name],
                                    path=save_path)

    @staticmethod
    def _create_train_objects(dataset: HOMRDataset,
                              batch_size: int
                              ) -> Tuple:
        """
        Creates the objects needed for the evaluating

        Args:
            dataset: Dataset used to feed the dataloaders
            batch_size: size of the batches in the train loader

        Returns: train loader

        """
        # Creation of the light dataset containing only the data and targets
        x, y, idx = dataset[dataset.train_mask]
        light_dataset = LightHOMRDataset(x, y, idx, dataset.temporal_analysis)

        # Creation of evaluating loader
        if dataset.temporal_analysis:
            sampler = BatchSampler(x, idx, batch_size=min(len(dataset.train_mask), batch_size))
            train_data = DataLoader(light_dataset,
                                    batch_size=1,
                                    batch_sampler=sampler,
                                    drop_last=False,
                                    shuffle=False)

        else:
            train_data = DataLoader(light_dataset, batch_size=min(len(dataset.train_mask), batch_size))


        return train_data

    @staticmethod
    def _validate_input_args(input_args: List[Any]) -> None:
        """
        Checks if all arguments related to inputs are None,
        if not the inputs are valid

        Args:
            input_args: list of arguments related to inputs

        Returns: None
        """
        valid = False
        for arg in input_args:
            if arg is not None:
                valid = True
                break

        if not valid:
            raise ValueError("There must be continuous columns_to_anonymize or categorical columns_to_anonymize")

    @staticmethod
    def _validate_sample_weights(dataset: HOMRDataset,
                                 sample_weights: Optional[tensor]) -> tensor:
        """
        Validates the provided sample weights and return them.
        If None are provided, each sample as the same weights of 1/n in the evaluating loss,
        where n is the number of elements in the dataset.

        Args:
            dataset: Dataset used to feed the dataloaders
            sample_weights: (N,) tensor with weights of the samples in the evaluating set

        Returns:

        """
        # We check the validity of the samples' weights
        dataset_size = len(dataset)
        if sample_weights is not None:
            if sample_weights.shape[0] != dataset_size:
                raise ValueError(f"sample_weights as length {sample_weights.shape[0]}"
                                 f" while dataset as length {dataset_size}")
        else:
            sample_weights = ones(dataset_size)  # / dataset_size

        return sample_weights

    @abstractmethod
    def _execute_train_step(self,
                            train_data: Tuple[DataLoader, HOMRDataset],
                            sample_weights: tensor) -> float:
        """
        Executes one evaluating epoch

        Args:
            train_data: evaluating dataloader or tuple (train loader, dataset)
            sample_weights: weights of the samples in the loss

        Returns: mean epoch loss
        """
        raise NotImplementedError

    @abstractmethod
    def _execute_valid_step(self,
                            valid_data: DataLoader,
                            sample_weights: tensor,
                            early_stopper: Optional[EarlyStopper]) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_data: valid dataloader or tuple (valid loader, dataset)
            early_stopper: early stopper keeping track of validation loss

        Returns: True if we need to early stop
        """
        raise NotImplementedError

    @staticmethod
    def disable_module_running_stats(module: Module) -> None:
        """
        Sets momentum to 0 for all BatchNorm layer in the module after saving it in a cache

        Args:
            module: torch module

        Returns: None
        """
        if isinstance(module, BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    @staticmethod
    def enable_module_running_stats(module: Module) -> None:
        """
        Restores momentum for all BatchNorm layer in the module using the value in the cache

        Args:
            module: torch module

        Returns: None
        """
        if isinstance(module, BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    def samples_weights(self, y):
        """
            Assigns samples weights
        """
        if self.weight is not None:
            n = len(y)
            w0, w1 = (1 - self.weight) / n, self.weight / n
            sample_weights = torch.zeros(y.shape)
            if len(torch.where(y == 0)) > 0:
                sample_weights[torch.where(y == 0)] = w0
            if len(torch.where(y == 1)) > 0:
                sample_weights[torch.where(y == 1)] = w1
        else:
            n = len(y)
            sample_weights = torch.zeros(y.shape)
            sample_weights[:] = 1 / n

        return sample_weights
