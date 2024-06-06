"""
Filename: ranger_forest.py

Authors: Hakima Laribi

Description: This file is used to define the classification
             wrappers for the skranger random forest models

"""

from typing import List, Optional, Callable

from skranger.ensemble import RangerForestClassifier

from src.models.wrappers.sklearn_wrappers import SklearnBinaryClassifierWrapper
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP


class HOMRBinaryRGFC(SklearnBinaryClassifierWrapper):
    """
    Skranger random forest classifier wrapper for the HOMR framework
    """
    def __init__(self,
                 n_estimators: int = 512,
                 mtry: int = 15,
                 min_node_size: int = 10,
                 max_depth: int = 0,
                 inbag: Optional[List] = None,
                 verbose: bool = False,
                 seed: int = 101,
                 classification_threshold: int = 0.5,
                 weight: float = None):
        """
        Creates a sklearn random forest classifier model and sets other protected
        attributes using parent's constructor

        Args:
            n_estimators: number of trees in the forest.
            mtry: the number of features to split on each node.
            min_node_size: the minimal node size.
            max_depth: the maximal tree depth; 0 means unlimited.
            inbag: a list of size 3 containing the parameters needed to call the inbag method which returns a list of
            size of n_estimators containing idx of observations to use for evaluating in each tree.
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to samples.
        """
        if inbag is not None:
            inbag_samples = inbag[0](inbag[1], n_estimators, seed, inbag[2])
        else:
            inbag_samples = None

        super().__init__(model=RangerForestClassifier(n_estimators=n_estimators,
                                                      mtry=mtry,
                                                      min_node_size=min_node_size,
                                                      max_depth=max_depth,
                                                      verbose=verbose,
                                                      inbag=inbag_samples,
                                                      seed=seed),
                         classification_threshold=classification_threshold,
                         weight=weight)

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(RangerForestHP()) + [RangerForestHP.WEIGHT]


class RangerForestHP:
    """
    Random forest's hyperparameters
    """
    MTRY = NumericalIntHP("mtry")
    MIN_NODE_SIZE = NumericalIntHP("min_node_size")
    MAX_DEPTH = NumericalIntHP("max_depth")
    N_ESTIMATORS = NumericalIntHP("n_estimators")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.MTRY, self.MIN_NODE_SIZE, self.MAX_DEPTH, self.N_ESTIMATORS])
