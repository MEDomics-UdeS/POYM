"""
Filename: lstm.py

Authors: Hakima Laribi

Description: This file is used to define the classification
             wrappers for LSTM models

"""

from typing import List, Optional

from src.models.abstract_models.lstm_base_models import LSTMBinaryClassifier
from src.models.wrappers.torch_wrappers import TorchBinaryClassifierWrapper
from src.utils.hyperparameters import CategoricalHP, HP, NumericalContinuousHP, NumericalIntHP
from src.utils.metric_scores import BinaryClassificationMetric, AUC


class HOMRBinaryLSTMC(TorchBinaryClassifierWrapper):
    """
    LSTM classification model wrapper for the HOMR framework
    """

    def __init__(self,
                 n_layer: int = 1,
                 n_unit: int = 10,
                 eval_metric: Optional[BinaryClassificationMetric] = AUC(),
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0.0001,
                 lr: float = 0.0001,
                 batch_size: int = 100,
                 valid_batch_size: int = 1000,
                 max_epochs: int = 200,
                 patience: int = 150,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 verbose: bool = True,
                 classification_threshold: float = 0.5,
                 model: str = 'LSTM',
                 bidirectional: bool = True,
                 weight: float = 0.87):
        """
        Builds a binary classification LSTM and sets the protected attributes using parent's constructor

        Args:
            n_layer: number of hidden layer
            n_unit: number of units in each hidden layer
            eval_metric: evaluation metric
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            batch_size: size of the batches in the evaluating loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: maximum number of epochs for evaluating
            patience: number of consecutive epochs without improvement
            num_cont_col: number of numerical continuous columns_to_anonymize
            cat_idx: idx of categorical columns_to_anonymize in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            verbose: if True, evaluating progress will be printed
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        # Model creation
        model = LSTMBinaryClassifier(layers=[n_unit] * n_layer,
                                     eval_metric=eval_metric,
                                     dropout=dropout,
                                     alpha=alpha,
                                     beta=beta,
                                     num_cont_col=num_cont_col,
                                     cat_idx=cat_idx,
                                     cat_sizes=cat_sizes,
                                     weight=weight,
                                     model=model,
                                     bidirectional=bidirectional,
                                     verbose=verbose)

        super().__init__(model=model,
                         classification_threshold=classification_threshold,
                         weight=weight,
                         train_params={'lr': lr,
                                       'batch_size': batch_size,
                                       'valid_batch_size': valid_batch_size,
                                       'patience': patience,
                                       'max_epochs': max_epochs})

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(RNNHP())

    def is_encoder(self) -> bool:
        """
        Checks if the model is an encoder

        Returns: boolean
        """
        return False


class RNNHP:
    """
    RNN's hyperparameters
    """
    ALPHA = NumericalContinuousHP("alpha")
    BATCH_SIZE = NumericalIntHP("batch_size")
    BETA = NumericalContinuousHP("beta")
    DROPOUT = NumericalContinuousHP("dropout")
    LR = NumericalContinuousHP("lr")
    N_LAYER = NumericalIntHP("n_layer")
    N_UNIT = NumericalIntHP("n_unit")
    WEIGHT = NumericalContinuousHP("weight")
    BIDIRECTIONAL = CategoricalHP("bidirectional")

    def __iter__(self):
        return iter([self.ALPHA, self.BATCH_SIZE,
                     self.BETA, self.LR, self.DROPOUT, self.N_LAYER, self.N_UNIT, self.WEIGHT, self.BIDIRECTIONAL])
