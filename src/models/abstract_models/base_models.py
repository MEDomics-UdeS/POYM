"""
Filename: base_models.py

Description: Defines the abstract HOMRBinaryClassifier classes that must be used
             to build every other model in the project.
             It ensures consistency will all hyperparameter tuning functions.

"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from numpy import array, argmax
from numpy import where as npwhere
from numpy import zeros as npzeros
from sklearn.metrics import roc_curve
from torch import tensor, is_tensor
from torch import where as thwhere
from torch import zeros as thzeros
from xgboost import XGBClassifier

from src.data.processing.datasets import HOMRDataset
from src.utils.hyperparameters import HP


class BinaryClassifier(ABC):
    """
    Skeleton of all binary classification models
    """

    def __init__(self,
                 classification_threshold: float = None,
                 weight: Optional[float] = None,
                 train_params: Optional[Dict[str, Any]] = None):
        """
        Sets the protected attributes of the object

        Args:
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1 (in [0, 1])
            train_params: keyword arguments that are proper to the child model inheriting
                          from this class and that will be using when calling fit method
        """
        if weight is not None:
            if not (0 <= weight <= 1):
                raise ValueError("weight must be included in range [0, 1]")

        self._thresh = classification_threshold
        self._train_params = train_params if train_params is not None else {}
        self._weight = weight

    @property
    def thresh(self) -> float:
        return self._thresh


    @property
    def train_params(self) -> Dict[str, Any]:
        return self._train_params

    @property
    def weight(self) -> Optional[float]:
        return self._weight

    def find_fixe_threshold(self,
                            pred: List,
                            n_positive_values: int = None) -> float:
        """
            Finds a fixed threshold according to the number of predicted as positive samples we want to consider
        """
        # if it is a pytorch model running in GPU we move all the variabels to the CPU
        if hasattr(self, '_model') and hasattr(self._model, '_device') and self._model._device == torch.device('cuda'):
            pred_cpu = pred.cpu()
        else:
            pred_cpu = pred

        if n_positive_values is not None:
            # Set the threshold according to the number of samples we want in the positive class
            return min(sorted(pred_cpu, reverse=True)[:n_positive_values])

    def find_optimal_threshold(self,
                               targets: List,
                               pred: List):
        """
            Update the optimal threshold found using the J function
        """
        # if it is a pytorch model running in GPU we move all the variabels to the CPU
        if hasattr(self, '_model') and hasattr(self._model, '_device') and self._model._device == torch.device('cuda'):
            targets_cpu = targets.cpu()
            pred_cpu = pred.cpu()
        else:
            targets_cpu = targets
            pred_cpu = pred

        # Find the optimal threshold
        self._thresh = self.optimize_J_statistic(targets_cpu, pred_cpu)

    def get_sample_weights(self, y_train: Union[tensor, array]) -> Union[tensor, array]:
        """
        Computes the weight associated to each sample

        We need to solve the following equation:
            - n1 * w1 = self.weight
            - n0 * w0 = 1 - self.weight

        where n0 is the number of samples with label 0
        and n1 is the number of samples with label 1

        Args:
            y_train: (N, 1) tensor or array with labels

        Returns: sample weights
        """
        # If no weight was provided we return None
        if self.weight is None:
            return None

        # Otherwise we return samples' weights in the appropriate format
        if hasattr(self, "_model") and isinstance(self._model, XGBClassifier):
            w0, w1 = (1 - self.weight), self.weight  # class weight for C0, class weight for C1
        else:
            n = len(y_train)
            w0, w1 = (1 - self.weight) / n, self.weight / n  # sample weight for C0, sample weight for C1

        # We save the weights in the appropriate format
        if not is_tensor(y_train):
            sample_weights = npzeros(y_train.shape)
            sample_weights[npwhere(y_train == 0)] = w0
            sample_weights[npwhere(y_train == 1)] = w1

        else:
            sample_weights = thzeros(y_train.shape)
            sample_weights[thwhere(y_train == 0)] = w0
            sample_weights[thwhere(y_train == 1)] = w1

        return sample_weights

    @staticmethod
    def optimize_J_statistic(targets: list,
                             pred: list) -> float:
        """
        Finds the optimal threshold from ROC curve that separates the negative and positive classes by optimizing
        the function the Youden's J statistics J = TruePositiveRate – FalsePositiveRate
        Args:
            targets: ground truth labels
            pred: predicted probabilities to belong to the positive class (OYM=1)

        Returns a float representing the optimal threshold
        """
        # Calculate roc curves
        fpr, tpr, thresholds = roc_curve(targets, pred, pos_label=1)

        # Get the best threshold
        J = tpr - fpr
        threshold = thresholds[argmax(J)]

        return threshold

    @staticmethod
    @abstractmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self,
            dataset: HOMRDataset) -> None:
        """
        Fits the model to the evaluating data

        Args:
            dataset: HOMRDataset which its items are tuples (x, y, idx) where
                           - x : (N,D) tensor or array with D-dimensional samples
                           - y : (N,) tensor or array with classification labels
                           - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self,
                      dataset: HOMRDataset,
                      mask: Optional[List[int]] = None) -> Union[tensor, array]:
        """
        Returns the probabilities of being in class 1 (OYM = 1) for all samples
        in a particular set (default = test)

        Args:
             dataset: HOMRDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) tensor or array
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self,
                   path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def is_encoder(self) -> bool:
        """
        Checks if the model is an encoder

        Returns: boolean
        """

        raise NotImplementedError
