"""
Filename: ensemble_lstm.py

Authors: Hakima Laribi

Description: This file is used to define the classification wrappers for ElSTM model

"""

import os
import pickle
from typing import List, Optional, Union

from numpy import array
from torch import tensor

from src.data.processing.datasets import HOMRDataset
from src.models.abstract_models.base_models import BinaryClassifier
from src.models.abstract_models.elstm_base_model import EnsembleLSTMBinaryClassifier
from src.utils.hyperparameters import HP


class HOMRBinaryELSTMC(BinaryClassifier):
    """
    ELSTM classifier model for the HOMR framework
    """

    def __init__(self,
                 pretrained_models: List[BinaryClassifier],
                 classification_threshold: float = 0.5,
                 weight: float = None):
        """
        Creates an ensemble LSTM classifier and sets protected attributes using parent's constructor

        Args:
            pretrained_models: list of all the pretrained models
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        # Model creation
        self._model = EnsembleLSTMBinaryClassifier(pretrained_models=pretrained_models)

        # Call of parent's constructor
        super().__init__(classification_threshold=classification_threshold,
                         weight=weight)

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return []

    def fit(self, dataset: HOMRDataset) -> None:
        """
        Fits the model to the evaluating data

        Args:
            dataset: HOMRDataset

        Returns: None
        """

        # We extract train set and eval set
        self._model.fit(dataset)

    def predict_proba(self,
                      dataset: HOMRDataset,
                      mask: Optional[List[int]] = None) -> Union[tensor, array]:
        """
        Returns the probabilities of being in class 1 (OYM = 1) for all samples
        in a particular set (default = test)

        Returns: (N,) tensor or array
        """

        return self._model.predict_proba(dataset, mask)

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
