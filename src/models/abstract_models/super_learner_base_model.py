from typing import Callable, List, Dict, Any, Union, Optional
import numpy as np
import torch

from src.data.processing.datasets import HOMRDataset
from src.models.abstract_models.base_models import BinaryClassifier


class SuperLearner:
    """
    Class that defines the Super Learner. This Super Learner uses a machine learning model to combine predictions
    of multiple models. For more details: https://doi.org/10.2202/1544-6115.1309
    """

    def __init__(self,
                 ml_model: Callable,
                 fixed_train_params: Dict[Any, Any],
                 pretrained_models_train: Dict[int, List[BinaryClassifier]],
                 pretrained_models_test: Dict[int, List[BinaryClassifier]]
                 ):
        """
        Sets public, protected and private attributes

        Args:
            ml_model: machine learning model used to combine the predictions of multiple models
            fixed_train_params: hyperparameters of the machine learning model
            pretrained_models_train: list of all the pretrained models for each split of the k-folds cross-validation
            pretrained_models_test: list of all the pretrained models on the entire learning set, used for final
            prediction on the holdout set.

        NOTE: the models used in the training procedure must remain the same in the final testing procedure, the
        models must also be passed in the same order.
        """
        # Validate inputs
        for split, split_pretrained_models in pretrained_models_train.items():
            if len(split_pretrained_models) != len(pretrained_models_test):
                raise ValueError(f'In split {split}, number of pretrained models {len(split_pretrained_models)} is '
                                 f'different from the number of pretrained models in the final testing procedure '
                                 f'{len(pretrained_models_test)}')

        # Sets protected attributes
        self._ml_model = ml_model(**fixed_train_params)
        self._pretrained_models_train = pretrained_models_train
        self._pretrained_models_test = pretrained_models_test

    def fit(self,
            dataset: HOMRDataset):
        """
        Fits the super learner which is a machine learning model on the probabilities predicted by
        each pretrained model
        """
        # Get the training dataset of the super learner : probabilities of out of fold samples for each split
        x, y = self.build_dataset(dataset, dataset.train_mask)
        # Train the super learner on the probabilities
        self._ml_model.fit(x, y)

    @staticmethod
    def build_dataset(dataset: HOMRDataset,
                      masks: Dict[int, List[List[int]]],
                      pretrained_models: Dict[int, List[BinaryClassifier]]):
        """
        Builds the dataset that represents the probabilities of each model for each individual in the mask

        Args:
            dataset: HOMR dataset
            masks : dictionary containing indexes masks for each split
            pretrained_models: pretrained models from which we record probabilities for each sample
        """
        x_probas = np.zeros((0, 0))
        y_probas = np.zeros(0)

        # Get probabilities for all out of folds samples for each split
        for split, mask in masks.items():
            x, y, _ = dataset[mask]
            x_split = np.zeros((0, 0))
            relative_indexes, _ = dataset.flatten_indexes(mask)
            # Get probabilities for each pretrained models on the current split
            for pretrained_model in pretrained_models[split]:
                probas = pretrained_model.predict_proba(dataset, mask).cpu().detach().numpy()
                x_split = np.hstack((x_split, probas[relative_indexes]))  # get the last index of each temporal sequence

            x_probas = np.vstack((x_probas, x_split))

            # Flatten y and get the last index of each temporal sequence
            y_probas = np.vstack((y_probas, torch.flatten(y).cpu().detach().numpy()[relative_indexes]))

        return x_probas, y_probas

    def predict_proba(self,
                      dataset: HOMRDataset,
                      mask: Optional[Dict[int, List[List[int]]]] = None) -> np.array:
        """
        Computes final probabilities using the Super Learner

        Args:
            dataset: HOMR dataset
            mask: dictionary with masks indexes
        """
        # Get mask
        mask = mask if mask is not None else dataset.test_mask

        # Get the pretrained models
        pretrained_models = self._pretrained_models_train if len(mask.keys()) > 1 else self._pretrained_models_test

        # We get the probabilities of pretrained models on the holdout set
        x, _ = self.build_dataset(dataset, mask, pretrained_models)

        # Predict probabilities
        return self._ml_model.predict_proba(x)
