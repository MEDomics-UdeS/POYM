"""
Filename: evaluation.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the Evaluator class in charge
             of comparing models against each other

Date of last modification : 2021/10/29
"""
from copy import deepcopy
from json import load
from os import makedirs, path
from time import strftime
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import pandas as pd
import ray
from numpy.random import seed as np_seed
import random
from torch import is_tensor, from_numpy, manual_seed
import pickle
from settings.paths import Paths
from src.data.processing import constants
from src.data.processing.datasets import HOMRDataset
from src.data.processing.sampling import MaskType
from src.models.abstract_models.base_models import BinaryClassifier
from src.recording.constants import PREDICTION, RECORDS_FILE, TEST_RESULTS, TRAIN_RESULTS, VALID_RESULTS
from src.recording.recording import Recorder, compare_prediction_recordings, \
    get_evaluation_recap, plot_hps_importance_chart
from src.evaluating.tuning import Objective, Tuner
from src.utils.metric_scores import Metric


class Evaluator:
    """
    Object in charge of evaluating a model over multiple different data splits.
    """

    def __init__(self,
                 model_constructor: Callable,
                 dataset: HOMRDataset,
                 masks: Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]],
                 hps: Dict[str, Dict[str, Any]],
                 n_trials: int,
                 evaluation_metrics: List[Metric],
                 fixed_params: Optional[Dict[str, Any]] = None,
                 seed: int = 101,
                 gpu_device: bool = False,
                 evaluation_name: Optional[str] = None,
                 fixed_params_update_function: Optional[Callable] = None,
                 save_hps_importance: Optional[bool] = True,
                 save_parallel_coordinates: Optional[bool] = True,
                 existing_eval: bool = False,
                 save_optimization_history: Optional[bool] = True,
                 train: str = None):
        """
        Set protected and public attributes

        Args:
            model_constructor: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluations
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            fixed_params: dictionary with parameters used by the model constructor for building model
            n_trials: number of hyperparameters sets sampled within each inner validation loop
            evaluation_metrics: list of metrics to evaluate on models built for each outer split.
                                The last one is used for hyperparameter optimization
            seed: random state used for reproducibility
            gpu_device: True if we want to use the gpu
            evaluation_name: name of the results file saved at the recordings_path
            fixed_params_update_function: function that updates fixed params dictionary from after feature
                                          selection. Might be necessary for model with entity embedding.
            save_hps_importance: true if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: true if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: true if we want to plot the optimization history graph after tuning
        """

        # We look if a file with the same evaluation name exists
        if evaluation_name is not None:
            if path.exists(path.join(Paths.EXPERIMENTS_RECORDS, evaluation_name)) & (not existing_eval):
                raise ValueError("evaluation with this name already exists")
        else:
            makedirs(Paths.EXPERIMENTS_RECORDS, exist_ok=True)
            evaluation_name = f"{strftime('%Y%m%d-%H%M%S')}"

        # We set protected attributes
        self._dataset = dataset
        self._gpu_device = gpu_device
        self._feature_selection_count = {feature: 0 for feature in self._dataset.original_data.columns}
        self._fixed_params = fixed_params if fixed_params is not None else {}
        self._hps = hps
        self._masks = masks
        self._hp_tuning = (n_trials > 0)
        self._tuner = Tuner(n_trials=n_trials,
                            save_hps_importance=save_hps_importance,
                            save_parallel_coordinates=save_parallel_coordinates,
                            save_optimization_history=save_optimization_history,
                            path=Paths.EXPERIMENTS_RECORDS)

        # We set the public attributes
        self.evaluation_name = evaluation_name
        self.model_constructor = model_constructor
        self.evaluation_metrics = evaluation_metrics
        self.seed = seed
        self.train = train
        # We set the fixed params update method
        if fixed_params_update_function is not None:
            self._update_fixed_params = fixed_params_update_function
        else:
            self._update_fixed_params = lambda _, a: self._fixed_params

    def evaluate(self) -> None:
        """
        Performs nested subsampling validations to evaluate a model and saves results
        in specific files using a recorder

        Returns: None

        """
        # We set the seed for the nested subsampling valid procedure
        if self.seed is not None:
            np_seed(self.seed)
            manual_seed(self.seed)
            random.seed(self.seed)

        # We execute the outer loop
        for k, v in self._masks.items():

            # We extract the masks
            train_mask, valid_mask = v[MaskType.TRAIN], v[MaskType.VALID]
            test_mask, in_masks = v[MaskType.TEST], v[MaskType.INNER]

            # Extract other test masks
            additional_test_masks = {}
            for (mask_type, idx_mask) in v.items():
                if mask_type not in [MaskType.TRAIN, MaskType.VALID, MaskType.TEST, MaskType.INNER]:
                    additional_test_masks[mask_type] = idx_mask

            # We update the dataset's masks
            self._dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

            # We create the Recorder object to save the result of this experience
            recorder = Recorder(evaluation_name=self.evaluation_name,
                                index=k,
                                recordings_path=Paths.EXPERIMENTS_RECORDS,
                                masks_types=list(v.keys()))

            # We save the saving path
            saving_path = path.join(Paths.EXPERIMENTS_RECORDS, self.evaluation_name, f"Split_{k}")

            # We proceed to feature selection
            subset = deepcopy(self._dataset)

            # We update the fixed parameters according to the subset
            self._fixed_params = self._update_fixed_params(subset, k)

            # Combine all the masks
            masks_names = {
                "train_set": train_mask,
                "valid_set": valid_mask,
                "test_set": test_mask,
                **{"test_" + name + "_set": mask for name, mask in additional_test_masks.items()}
            }

            # Record the data count
            for name, mask in masks_names.items():
                if mask is None:
                    mask_length = 0
                else:
                    mask_length = sum(len(idx) for idx in mask.values()) if isinstance(mask, dict) else len(mask)
                recorder.record_data_info(name, mask_length)

            # We update the tuner to perform the hyperparameters optimization
            if self._hp_tuning:

                print(f"\nHyperparameter tuning started - K = {k}\n")

                # We update the tuner
                self._tuner.update_tuner(study_name=f"{self.evaluation_name}_{k}",
                                         objective=self._create_objective(masks=in_masks, subset=subset,
                                                                          update_params=self._update_fixed_params),
                                         saving_path=saving_path)

                # We perform the hps tuning to get the best hps
                best_hps, hps_importance = self._tuner.tune()

                # We save the hyperparameters
                print(f"\nHyperparameter tuning done - K = {k}\n")
                recorder.record_hyperparameters(best_hps)

                # We save the hyperparameters importance
                recorder.record_hyperparameters_importance(hps_importance)
            else:
                best_hps = {}

            # We create a model with the best hps
            model = self.model_constructor(**best_hps, **self._fixed_params)

            # We train our model with the best hps
            print(f"\nFinal model training - K = {k}\n")
            if self._hp_tuning:
                subset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            if self.train is None:
                model.fit(dataset=subset)
                # We save plots associated to evaluating
                if hasattr(model, 'plot_evaluations') and valid_mask is not None:
                    model.plot_evaluations(save_path=saving_path)

                # We save the trained model
                model.save_model(path=saving_path)
            else:
                model._model = torch.load(self.train)

            # We get the predictions and save the evaluation metric scores
            self._record_scores_and_pred(model, recorder, subset, additional_test_masks)

            # We save all the data collected in one file
            recorder.generate_file()

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=self.evaluation_name,
                             recordings_path=Paths.EXPERIMENTS_RECORDS,
                             masks_types=list(v.keys()))

        # We save the hyperparameters importance chart
        try:
            plot_hps_importance_chart(evaluation_name=self.evaluation_name,
                                      recordings_path=Paths.EXPERIMENTS_RECORDS)
        except:
            pass

    def _create_objective(self,
                          masks: Dict[int, Dict[str, List[int]]],
                          subset: HOMRDataset,
                          update_params: Callable,
                          ) -> Objective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning
            subset: subset of the original dataset after feature selection

        Returns: objective function
        """

        metric = self.evaluation_metrics[-1] if len(self.evaluation_metrics) > 0 else None

        return Objective(dataset=subset,
                         masks=masks,
                         hps=self._hps,
                         fixed_params=self._fixed_params,
                         metric=metric,
                         model_constructor=self.model_constructor,
                         update_params=update_params,
                         gpu_device=self._gpu_device)

    def _record_scores_and_pred(self,
                                model: BinaryClassifier,
                                recorder: Recorder,
                                subset: HOMRDataset,
                                additional_test_masks: Dict[str, List[int]]) -> None:
        """
        Records the scores associated to train and test set
        and also saves the prediction linked to each individual

        Args:
            model: model trained with best found hyperparameters
            recorder: object recording information about splits evaluations
            subset: dataset with remaining features from feature selection

        Returns: None
        """

        # We find the optimal threshold and save it
        pred = model.predict_proba(subset, subset.train_mask)

        # Get the train indexes
        indexes = (
            [idx for idx_split in subset.train_mask.values() for idx in idx_split]
            if isinstance(subset.train_mask, dict)
            else subset.train_mask
        )

        # Get the training targets
        _, y_train, _ = subset[indexes]

        if subset.temporal_analysis:
            # Put the targets in the right format
            y_train = torch.cat(y_train)

        # Get the indexes of each last element of a sequence
        relative_indexes, _ = subset.flatten_indexes(indexes)
        # Get the optimal threshold on these data
        model.find_optimal_threshold(y_train[relative_indexes], pred[relative_indexes])

        recorder.record_data_info('thresh', str(model.thresh))

        m = [tuple(zip(additional_test_masks.values(), additional_test_masks.keys()))[i]
             for i in range(len(additional_test_masks.values()))]

        mask_list = [(subset.train_mask, MaskType.TRAIN),
                     (subset.valid_mask, MaskType.VALID),
                     (subset.test_mask, MaskType.TEST)] + m

        for mask, mask_type in mask_list:
            if len(mask) > 0:
                pred = model.predict_proba(subset, mask)

                # Get the indexes
                mask = (
                    [idx for idx_split in subset.train_mask.values() for idx in idx_split]
                    if isinstance(subset.train_mask, dict)
                    else subset.train_mask
                )

                # We extract targets
                _, y, _ = subset[mask]

                if subset.temporal_analysis:
                    # Get relative indexes
                    relative_indexes, flatten_mask = subset.flatten_indexes(mask)

                    # Get last element of each sequence
                    y = torch.cat(y)[relative_indexes]
                    pred = pred[relative_indexes]
                    mask = flatten_mask

                # We extract ids for recording
                ids = [subset.ids[i] for i in mask]

                threshold = model.thresh
                # We record all metric scores
                for metric in self.evaluation_metrics:
                    if torch.is_tensor(y) and not torch.is_tensor(pred):
                        recorder.record_scores(score=metric(pred, y.cpu().detach().numpy(), thresh=threshold),
                                               metric=metric.name, mask_type=mask_type)
                    else:
                        recorder.record_scores(score=metric(pred, y, thresh=threshold),
                                               metric=metric.name, mask_type=mask_type)

                if not is_tensor(pred):
                    pred = from_numpy(pred)

                # We get the final predictions from the soft predictions
                proba = pred
                pred = (pred >= threshold).long()

                # We save the predictions
                recorder.record_predictions(probabilities=proba, predictions=pred, ids=ids, targets=y,
                                            mask_type=mask_type)

    def _load_predictions(self,
                          split_number: int,
                          subset: pd.DataFrame) -> pd.DataFrame:
        """""
        Loads prediction in a given path and includes them as a feature within the dataset

        Args:
            split_number: split for which we got to load predictions
            subset: actual dataset

        Returns: updated dataset
        """

        # Loading of records
        with open(path.join(self._pred_path, f"Split_{split_number}", RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        # We check the format of predictions
        random_pred = list(data[TRAIN_RESULTS].values())[0][PREDICTION]
        if "[" not in random_pred:

            # Saving of the number of predictions columns
            nb_pred_col = 1

            # Creation of the conversion function to extract predictions from strings
            def convert(x: str) -> List[float]:
                return [float(x)]
        else:

            # Saving of the number of predictions columns
            nb_pred_col = len(random_pred[1:-1].split(','))

            # Creation of the conversion function to extract predictions from strings
            def convert(x: str) -> List[float]:
                return [float(a) for a in x[1:-1].split(',')]

        # Extraction of predictions
        pred = {}
        for section in TRAIN_RESULTS, TEST_RESULTS, VALID_RESULTS:
            pred = {**pred, **{p_id: [p_id, *convert(v[PREDICTION])] for p_id, v in data[section].items()}}

        # Creation of pandas dataframe
        pred_col_names = [f'pred{i}' for i in range(nb_pred_col)]
        df = pd.DataFrame.from_dict(pred, orient='index', columns=[constants.IDS, *pred_col_names])

        # Creation of new augmented dataset
        return subset.create_superset(data=df, categorical=False)
