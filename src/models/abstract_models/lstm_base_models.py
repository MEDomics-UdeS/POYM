"""
Filename: lstm_base_models.py

Authors: Hakima Laribi

Description: This file is used to define the LSTM model

"""

from typing import Callable, List, Optional, Tuple

import torch
from torch import cat, no_grad, tensor, sigmoid, stack
from torch.nn import BCEWithLogitsLoss, Linear, ReLU, Dropout
from torch.utils.data import DataLoader

from src.data.processing.datasets import HOMRDataset
from src.data.processing.sampling import MaskType
from src.evaluating.early_stopping import EarlyStopper
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.utils.metric_scores import BinaryCrossEntropy, Metric


class LSTM(TorchCustomModel):
    """
    LSTM model
    """

    def __init__(self,
                 output_size: int,
                 layers: List[int],
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 model: str = 'LSTM',
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 bidirectional: bool = True,
                 verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            output_size: the number of nodes in the last layer of the neural network
            layers: list with number of units in each hidden layer
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns_to_anonymize in the dataset
            cat_idx: idx of categorical columns_to_anonymize in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            verbose: True if we want trace of the evaluating progress
        """

        # We call parent's constructor
        super().__init__(criterion=criterion,
                         criterion_name=criterion_name,
                         eval_metric=eval_metric,
                         output_size=output_size,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         verbose=verbose)
        if model == 'LSTM':
            self.rnn_model = torch.nn.LSTM
        elif model == 'GRU':
            self.rnn_model = torch.nn.GRU
        else:
            self.rnn_model = torch.nn.RNN
        self.rnn_block = self.rnn_model(input_size=self._input_size,
                                        hidden_size=layers[-1],
                                        num_layers=len(layers),
                                        bidirectional=bidirectional,
                                        batch_first=True)

        # We add a linear layer to complete the layers
        if bidirectional:
            self._linear_layer = Linear(layers[-1] * 2, 2)
            self._linear_layer2 = Linear(2, output_size)
        else:
            self._linear_layer = Linear(layers[-1], 2)
            self._linear_layer2 = Linear(2, output_size)
        self.relu = ReLU()
        self.dropout = Dropout(p=dropout)

    def _execute_train_step(self,
                            train_data: DataLoader,
                            sample_weights: tensor) -> Tuple[float, float]:
        """
        Executes one evaluating epoch

        Args:
            train_data: evaluating dataloader
            sample_weights: weights of the samples in the loss

        Returns: mean epoch loss
        """
        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0
        y_val, p_val = torch.empty(0), torch.empty(0)
        nb_batch = 0
        # We execute one evaluating step
        for item in train_data:

            # We extract the data
            x, y, idx = item
            nb_batch += 1
            # Put the data in the appropriate shape
            if isinstance(y, list):
                y = cat(y)
                x = stack(x)
            y = torch.flatten(y.to(self._device))

            idx = torch.stack(idx).transpose(0, 1)

            # We clear the gradients
            self._optimizer.zero_grad()

            # Put the data in the appropriate device
            x_, y = [xi.to(self._device) for xi in [x]], y.to(self._device)
            x = x_

            # We perform the weight update
            y, pred, loss = self._update_weights(sample_weights, x, y, idx)

            # We update the metrics history
            epoch_loss += loss
            y_val = torch.cat((y_val, y))
            p_val = torch.cat((p_val, pred))

        # We save mean epoch loss and mean epoch score
        mean_epoch_loss = self.loss(sample_weights, p_val, y_val).item()
        mean_epoch_score = self._eval_metric(sigmoid(p_val).cpu(), y_val)
        self._evaluations[MaskType.TRAIN][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.TRAIN][self._eval_metric.name].append(mean_epoch_score)

        return mean_epoch_loss, mean_epoch_score

    def _execute_valid_step(self,
                            valid_loader: Optional[DataLoader],
                            sample_weights: tensor,
                            early_stopper: EarlyStopper) -> Tuple[bool, float, float]:
        """
        Executes an inference step on the validation data

        Args:
            valid_loader: validation dataloader
            early_stopper: early stopper keeping track of the validation loss

        Returns: True if we need to early stop
        """
        if valid_loader is None:
            return False, 0, 0

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0
        y_val, p_val = torch.empty(0), torch.empty(0)
        nb_batch = 0
        # We execute one inference step on validation set
        with no_grad():

            for item in valid_loader:

                # We extract the data
                x, y, idx = item
                nb_batch += 1
                # Put the data in the appropriate shape
                if isinstance(y, list):
                    y = cat(y)
                    x = stack(x)
                y = torch.flatten(y.to(self._device))

                idx = torch.stack(idx).transpose(0, 1)

                # We perform the forward pass
                output = self(x).squeeze(dim=-1)

                # Put the data in the appropriate device
                y = y.to(self._device)
                output = output.to(self._device)

                flatten_mask = []
                relative_indexes = []
                initial_index = 0
                for i, indexes in enumerate(idx):
                    flatten_mask.append(indexes[-1])
                    relative_indexes.append(initial_index + len(indexes) - 1)
                    initial_index += len(indexes)

                y = y[relative_indexes]
                output = output[relative_indexes]

                # We calculate the loss and the score
                # Sample weights are based on the ground truth but no data leakage since not used
                # in training/updating just for visualizing
                epoch_loss += self.loss(sample_weights, output, y).item()
                y_val = torch.cat((y_val, y))
                p_val = torch.cat((p_val, output))

        # We save mean epoch loss and mean epoch score
        mean_epoch_loss = self.loss(sample_weights, p_val, y_val).item()
        mean_epoch_score = self._eval_metric(sigmoid(p_val).cpu(), y_val)
        self._evaluations[MaskType.VALID][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.VALID][self._eval_metric.name].append(mean_epoch_score)

        # We check early stopping status
        early_stopper(mean_epoch_score, self)

        if early_stopper.early_stop:
            return True, mean_epoch_loss, mean_epoch_score

        return False, mean_epoch_loss, mean_epoch_score

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with values of the node within the last layer

        """
        # We initialize a list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, :, self._cont_idx])

        # We concatenate all inputs
        new_x_ = [xi.to(self._device) for xi in new_x]
        x = cat(new_x_, 1)

        # Put data in the appropriate device
        x = x.to(self._device)

        output, _, = self.rnn_block(x.to(torch.float32))
        output = self._linear_layer(self.dropout(self.relu(output.flatten(0, 1))))
        output = self._linear_layer2(output)
        return output


class LSTMBinaryClassifier(LSTM):
    """
    Multilayer perceptron binary classification model with entity embedding
    """

    def __init__(self,
                 layers: List[int],
                 eval_metric: Optional[Metric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 model: str = 'LSTM',
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 weight=None,
                 bidirectional: bool = True,
                 verbose: bool = False):
        """
        Sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns_to_anonymize
            cat_idx: idx of categorical columns_to_anonymize in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            verbose: true to print evaluating progress when fit is called
        """
        eval_metric = eval_metric if eval_metric is not None else BinaryCrossEntropy()
        self.weight = weight
        super().__init__(output_size=1,
                         layers=layers,
                         criterion=BCEWithLogitsLoss(reduction='none'),
                         criterion_name='WBCE',
                         eval_metric=eval_metric,
                         dropout=dropout,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         model=model,
                         bidirectional=bidirectional,
                         verbose=verbose)

    def predict_proba(self,
                      dataset: HOMRDataset,
                      mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the probabilities of being in class 1 (OYM = 1) for all samples
        in a particular set (default = test)

        Args:
             dataset: HOMRDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) tensor
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a sigmoid
        with no_grad():
            return torch.cat([sigmoid(self(x[i].unsqueeze(dim=0))).cpu() for i in range(len(x))], dim=0).squeeze()
