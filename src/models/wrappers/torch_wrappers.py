"""

Filename: torch_wrappers.py

Author: Nicolas Raymond
        Hakima Laribi

Description: This file is used to define the abstract classes
             used as wrappers for custom torch models
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from torch import tensor, save

from src.data.processing.datasets import HOMRDataset
from src.models.abstract_models.base_models import BinaryClassifier


class TorchBinaryClassifierWrapper(BinaryClassifier):
    """
    Class used as a wrapper for binary classifier inheriting from TorchCustomModel
    """

    def __init__(self,
                 model: Callable,
                 classification_threshold: float = 0.5,
                 weight: float = 0.5,
                 train_params: Optional[Dict[str, Any]] = None):

        """
        Sets the model protected attribute and other protected attributes via parent's constructor

        Args:
            model: classification model inheriting from TorchCustomModel
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            train_params: evaluating parameters proper to model for fit function
        """
        self._model = model
        super().__init__(classification_threshold=classification_threshold,
                         weight=weight,
                         train_params=train_params)

    @property
    def model(self) -> Callable:
        return self._model

    def fit(self, dataset: Union[HOMRDataset, Tuple[HOMRDataset, tensor, tensor]]) -> None:
        """
        Fits the model to the evaluating data

        Args:
            dataset: HOMRDataset which its items are tuples (x, y, idx) where
                           - x : (N,D) tensor or array with D-dimensional samples
                           - y : (N,) tensor or array with classification labels
                           - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """

        # We extract all labels
        _, y_train, _ = dataset[list(range(len(dataset)))]

        # Store the model in the appropriate device
        self._model = self._model.to(self._model._device)

        # Call the fit method
        self._model.fit(dataset, sample_weights=None, **self.train_params)

    def plot_evaluations(self, save_path: Optional[str] = None) -> None:
        """
        Plots the evaluating and valid curves saved

        Args:
            save_path: path where the figures will be saved

        Returns: None
        """
        if hasattr(self._model, 'plot_evaluations'):
            self._model.plot_evaluations(save_path=save_path)

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

        # Call predict_proba method, takes the prediction for class 1 and squeeze the array
        proba = self._model.predict_proba(dataset, mask)

        if isinstance(proba, tuple):
            proba, embeddings = proba
            return proba.squeeze(), embeddings
        return proba.squeeze()

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """

        save(self._model, os.path.join(path, "torch_model.pt"))
