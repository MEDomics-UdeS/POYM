"""
Filename: tabnet.py

Authors: Nicolas Raymond

Description: This file is used to define the regression and classification
             wrappers TabNet models

Date of last modification : 2021/10/25
"""


import os
from typing import List, Optional, Union

from numpy import array, ones
from pytorch_tabnet.tab_model import TabNetClassifier
from torch import tensor

from src.data.datasets import HOMRDataset
from src.models.abstract_models.base_models import HOMRBinaryClassifier
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from src.utils.visualization import visualize_epoch_progression


class HOMRBinaryTNC(HOMRBinaryClassifier):
    """
    TabNet classifier model wrapper for the HOMR framework
    """
    def __init__(self,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 lr: float = 0.001,
                 beta: float = 0.001,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 batch_size: int = 10000,
                 max_epochs: int = 200,
                 patience: int = 15,
                 device: str = 'cuda',
                 verbose: bool = False,
                 classification_threshold: float = 0.5,
                 weight: float = 0.5):
        """
        Creates a TabNet classifier and sets protected attributes using parent's constructor

        Args:
            n_d: width of the decision prediction layer. Bigger values gives more capacity to the model
                 with the risk of overfitting. Values typically range from 8 to 64.
            n_a: width of the attention embedding for each mask. According to the paper
                 n_d=n_a is usually a good choice.
            n_steps: number of steps in the architecture (usually between 3 and 10)
            gamma: coefficient for feature reusage in the masks. A value close to 1 will make
                   mask selection least correlated between layers. Values range from 1.0 to 2.0.
            lr: learning rate
            beta: L2 penalty coefficient
            cat_idx: list of categorical features indices.
            cat_sizes: list of categorical features number of modalities
            cat_emb_sizes: list of embeddings size for each categorical features.
            batch_size: number of samples per batch.
            max_epochs: maximum number of epochs for evaluating.
            patience: number of consecutive epochs without improvement
            device: 'cpu' or 'gpu'
            verbose: true to show evaluating loss progression
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        if cat_idx is None:
            cat_idx, cat_sizes, cat_emb_sizes = [], [], 1

        # Model creation
        self.__model = TabNetClassifier(n_d=n_d,
                                        n_a=n_a,
                                        n_steps=n_steps,
                                        gamma=gamma,
                                        cat_idxs=cat_idx,
                                        cat_dims=cat_sizes,
                                        cat_emb_dim=cat_emb_sizes,
                                        device_name=device,
                                        optimizer_params=dict(lr=lr, weight_decay=beta),
                                        verbose=int(verbose))

        # Call of parent's constructor
        super().__init__(classification_threshold=classification_threshold,
                         weight=weight,
                         train_params={'batch_size': batch_size,
                                       'max_epochs': max_epochs,
                                       'patience': patience})

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(TabNetHP()) + [TabNetHP.WEIGHT]

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

        # We extract train set and eval set
        x_train, y_train, _ = dataset[dataset.train_mask]
        eval_set = None

        if len(dataset.valid_mask) != 0:
            x_valid, y_valid, _ = dataset[dataset.valid_mask]
            eval_set = [(x_valid, y_valid)]

        # We get sample weights
        sample_weights = self.get_sample_weights(y_train)
        if sample_weights is None:
            sample_weights = ones(len(y_train))/len(y_train)

        self.__model.fit(x_train, y_train, weights=sample_weights,
                         eval_set=eval_set, eval_metric=["logloss"],
                         **self.train_params)

    def plot_evaluations(self, save_path: Optional[str] = None) -> None:
        """
        Plots the evaluating and valid curves saved

        Args:
            save_path: path where the figures will be saved

        Returns: None
        """
        # Extraction of data
        train_loss = self.__model.history['loss']
        valid_loss = self.__model.history['val_0_logloss']

        # Figure construction
        visualize_epoch_progression(train_history=[train_loss],
                                    valid_history=[valid_loss],
                                    progression_type=['WBCE'],
                                    path=save_path)

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

        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        return self.__model.predict_proba(x)[:, 1]

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        self.__model.save_model(os.path.join(path, "model"))

    def is_encoder(self) -> bool:
        """
        Checks if the model is an encoder

        Returns: boolean
        """
        return False


class TabNetHP:
    """
    TabNet's hyperparameters
    """
    BATCH_SIZE = NumericalIntHP("batch_size")
    BETA = NumericalContinuousHP("beta")
    GAMMA = NumericalContinuousHP("gamma")
    N_A = NumericalIntHP("n_a")
    N_D = NumericalIntHP("n_d")
    N_STEPS = NumericalIntHP("n_steps")
    LR = NumericalContinuousHP("lr")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.BATCH_SIZE, self.BETA, self.GAMMA, self.N_A, self.N_D, self.N_STEPS, self.LR])
