"""

Filename: sklearn_wrappers.py

Author: Nicolas Raymond
        Hakima Laribi

Description: This file is used to define the abstract classes
             used as wrappers for models with the sklearn API

"""
import os
import pickle
from typing import Any, Callable, List, Dict, Optional

import pandas as pd
from numpy import array
from xgboost import XGBClassifier
import torch
from src.data.processing.datasets import HOMRDataset
from src.models.abstract_models.base_models import BinaryClassifier


class SklearnBinaryClassifierWrapper(BinaryClassifier):
    """
    Class used as a wrapper for binary classifier with sklearn API
    """
    def __init__(self,
                 model: Callable,
                 classification_threshold: float = None,
                 weight: Optional[float] = None,
                 train_params: Optional[Dict[str, Any]] = None):

        """
        Sets the model protected attribute and other protected attributes via parent's constructor

        Args:
            model: classification model with sklearn API
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            train_params: evaluating parameters proper to model for fit function
        """
        self._model = model
        super().__init__(classification_threshold=classification_threshold,
                         weight=weight,
                         train_params=train_params)

    def fit(self, dataset: HOMRDataset) -> None:
        """
        Fits the model to the evaluating data

        Args:
            dataset: HOMRDataset which its items are tuples (x, y, idx) where
                           - x : (N,D) tensor or array with D-dimensional samples
                           - y : (N,) tensor or array with classification labels
                           - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """
        # We extract train set
        x_train, y_train, _ = dataset[dataset.train_mask]
        eval_set = None

        # We get the sample weights
        sample_weights = self.get_sample_weights(y_train)

        if len(dataset.valid_mask) != 0:
            x_valid, y_valid, _ = dataset[dataset.valid_mask]
            if torch.is_tensor(x_train):
                eval_set = [(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy()),
                            (x_valid.cpu().detach().numpy(), y_valid.cpu().detach().numpy())]
            else:
                eval_set = [(x_train, y_train), (x_valid, y_valid)]

        # Call the fit method
        if isinstance(self._model, XGBClassifier):
            if torch.is_tensor(x_train):
                self._model.fit(x_train.cpu().detach().numpy(),
                                y_train.cpu().detach().numpy(),
                                sample_weight=sample_weights.cpu().detach().numpy(),
                                eval_set=eval_set,
                                **self.train_params)
            else:
                self._model.fit(x_train,
                                y_train,
                                sample_weight=sample_weights,
                                eval_set=eval_set,
                                **self.train_params)
        else:
            self._model.fit(x_train, y_train, sample_weight=sample_weights, **self.train_params)

    def predict_proba(self,
                      dataset: pd.DataFrame,
                      mask: List[int] = None) -> array:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: HOMR dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) tensor or array
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Call predict_proba method and takes the prediction for class 1
        if torch.is_tensor(x):
            proba = self._model.predict_proba(x.cpu().detach().numpy())[:, 1]
        else:
            if isinstance(self._model, XGBClassifier):
                proba = self._model.predict_proba(x, ntree_limit=self._model.best_ntree_limit)[:, 1]
            else:
                proba = self._model.predict_proba(x)[:, 1]

        return proba.squeeze()

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        # We save the model with pickle
        filepath = os.path.join(path, "sklearn_model.sav")
        pickle.dump(self._model, open(filepath, "wb"))

    def is_encoder(self) -> bool:
        """
        Checks if the model is an encoder

        Returns: boolean
        """
        return False


