"""
Filename: metric_scores.py

Authors: Hakima Laribi
         Nicolas Raymond

Description: This file is used to define binary classifications metrics

"""


from abc import ABC, abstractmethod
from typing import Tuple, Union

from numpy import array
from sklearn.metrics import brier_score_loss, auc, roc_curve, precision_recall_curve
from torch import from_numpy, is_tensor, mean, pow, prod, tensor, zeros
from torch.nn.functional import binary_cross_entropy_with_logits


class Direction:
    """
    Custom enum for optimization directions
    """
    MAXIMIZE: str = "maximize"
    MINIMIZE: str = "minimize"

    def __iter__(self):
        return iter([self.MAXIMIZE, self.MINIMIZE])


class Reduction:
    """
    Custom enum for metric reduction choices
    """
    MEAN: str = "mean"
    SUM: str = "sum"
    GEO_MEAN: str = "geometric_mean"


class TaskType:
    """
    Custom enum for task types
    """
    REG: str = "regression"
    CLASSIFICATION: str = "classification"

    def __iter__(self):
        return iter([self.REG, self.CLASSIFICATION])


class Metric(ABC):
    """
    Abstract class that represents the skeleton of callable classes to use as optimization metrics
    """
    def __init__(self,
                 direction: str,
                 name: str,
                 task_type: str = TaskType.CLASSIFICATION,
                 n_digits: int = 5):
        """
        Sets protected attributes

        Args:
            direction: "maximize" or "minimize"
            name: name of the metric
            task_type: "regression" or "classification"
            n_digits: number of digits kept
        """
        if direction not in Direction():
            raise ValueError("direction must be in {'maximize', 'minimize'}")

        if task_type not in TaskType():
            raise ValueError("task_type must be in {'regression', 'classification'}")

        # Protected attributes
        self._direction = direction
        self._name = name
        self._task_type = task_type
        self._n_digits = n_digits

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_digits(self) -> int:
        return self._n_digits

    @abstractmethod
    def convert_to_tensors(self,
                           pred: Union[array, tensor],
                           targets: Union[array, tensor]) -> Tuple[tensor, tensor]:
        """
        Converts inputs to tensors

        Args:
            pred: (N,) tensor or array with predictions
            targets: (N,) tensor or array with ground truth

        Returns: rounded metric score
        """
        raise NotImplementedError


class BinaryClassificationMetric(Metric):
    """
    Abstract class that represents the skeleton of callable classes to use as classification metrics
    """
    def __init__(self,
                 direction: str,
                 name: str,
                 n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            direction: "maximize" or "minimize"
            name: name of the metric
            n_digits: number of digits kept
        """
        super().__init__(direction=direction, name=name, task_type=TaskType.CLASSIFICATION, n_digits=n_digits)

    def __call__(self,
                 pred: Union[array, tensor],
                 targets: Union[array, tensor],
                 thresh: float = 0.5) -> float:
        """
        Converts inputs to tensors, applies softmax if shape is different than expected
        and than computes the metric and applies rounding

        Args:
            pred: (N,) tensor or array with predicted probabilities of being in class 1

            targets: (N,) tensor or array with ground truth

        Returns: rounded metric score
        """

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)
        try:
            return round(self.compute_metric(pred, targets, thresh), self.n_digits)
        except ZeroDivisionError:
            return -1.

    def convert_to_tensors(self,
                           pred: Union[array, tensor],
                           targets: Union[array, tensor]) -> Tuple[tensor, tensor]:
        """
        Converts predictions to float (since they are probabilities) and ground truth to long

        Args:
            pred: (N,) tensor with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth

        Returns: (N,) tensor, (N,) tensor

        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).long()
        else:
            return pred, targets

    @staticmethod
    def get_confusion_matrix(pred_proba: tensor,
                             targets: tensor,
                             thresh: float) -> tensor:
        """
        Returns the confusion matrix

        Args:
            pred_proba: (N,) tensor with with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: (2,2) tensor

        """
        # We initialize an empty confusion matrix
        conf_matrix = zeros(2, 2)

        # We fill the confusion matrix
        pred_labels = (pred_proba >= thresh).long()

        for t, p in zip(targets, pred_labels):
            conf_matrix[t, p] += 1

        return conf_matrix

    @abstractmethod
    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the metric score

        Args:
            pred: (N,) tensor with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: metric score
        """
        raise NotImplementedError


class AUC(BinaryClassificationMetric):
    """
    Callable class that computes the AUC for ROC curve
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="AUC", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float = 0.5) -> float:
        """
        Returns the AUC for ROC curve

        Args:
            pred: (N,) tensor with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1
                    (Not used here)

        Returns: float
        """
        fpr, tpr, thresholds = roc_curve(targets.detach().numpy(), pred.detach().numpy(), pos_label=1)
        return auc(fpr, tpr)


class AUPRC(BinaryClassificationMetric):
    """
    Callable class that computes the AUC for ROC curve
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="AUPRC", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float = 0.5) -> float:
        """
        Returns the AUC for ROC curve

        Args:
            pred: (N,) tensor with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1
                    (Not used here)

        Returns: float
        """
        precision, recall, _ = precision_recall_curve(targets, pred, pos_label=1)
        return auc(recall, precision)


class BrierScore(BinaryClassificationMetric):
    """
    Callable class that computes the AUC for ROC curve
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MINIMIZE, name="BrierScore", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float = 0.5) -> float:
        """
        Returns the AUC for ROC curve

        Args:
            pred: (N,) tensor with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1
                    (Not used here)

        Returns: float
        """
        return brier_score_loss(targets, pred, pos_label=1)


class BinaryAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes the accuracy
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="Accuracy", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float = 0.5) -> float:
        """
        Returns the accuracy of predictions, according to the threshold

        Args:
            pred: (N,) tensor with predicted probabilities of being in class 1
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """
        pred_labels = (pred >= thresh).float()
        return (pred_labels == targets).float().mean().item()


class BinaryBalancedAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes balanced accuracy using confusion matrix
    """
    def __init__(self,
                 reduction: str = Reduction.MEAN,
                 n_digits: int = 5):
        """
         Sets the protected reduction method and other protected attributes using parent's constructor

        Args:
            reduction: "mean" for (TPR + TNR)/2 or "geometric_mean" for sqrt(TPR*TNR)
            n_digits: number of digits kept for the score
        """
        if reduction not in [Reduction.MEAN, Reduction.GEO_MEAN]:
            raise ValueError(f"Reduction must be in {[Reduction.MEAN, Reduction.GEO_MEAN]}")

        if reduction == Reduction.MEAN:
            self._reduction = mean
            name = "BalancedAcc"
        else:
            self._reduction = lambda x: pow(prod(x), exponent=(1/x.shape[0]))
            name = "GeoBalancedAcc"

        super().__init__(direction=Direction.MAXIMIZE, name=name, n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float = 0.5) -> float:
        """
        Returns the either (TPR + TNR)/2 or sqrt(TPR*TNR)

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """
        # We get confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We get TNR and TPR
        correct_rates = conf_mat.diag() / conf_mat.sum(dim=1)

        return self._reduction(correct_rates).item()


class BinaryCrossEntropy(BinaryClassificationMetric):
    """
    Callable class that computes binary cross entropy
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MINIMIZE, name="BCE", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float = 0.5) -> float:
        """
        Computes binary cross entropy using loss from pytorch

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """
        return binary_cross_entropy_with_logits(pred, targets.float()).item()


class BalancedAccuracyEntropyRatio(BinaryClassificationMetric):
    """
    Callable class that computes the ratio between binary balanced accuracy and binary cross entropy
    """
    def __init__(self,
                 reduction: str = Reduction.MEAN,
                 n_digits: int = 5):
        """
         Builds two metric and sets other protected attributes using parent's constructor

        Args:
            reduction: "mean" for (TPR + TNR)/2 or "geometric_mean" for sqrt(TPR*TNR)
            n_digits: number of digits kept for the score
        """
        self.bce = BinaryCrossEntropy(n_digits=10)
        self.bbacc = BinaryBalancedAccuracy(reduction=reduction, n_digits=10)
        super().__init__(direction=Direction.MAXIMIZE, name=f"{self.bbacc.name}/{self.bce.name}", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the ratio between binary balanced accuracy and binary cross entropy

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """
        return self.bbacc(pred, targets, thresh)/self.bce(pred, targets)


class Sensitivity(BinaryClassificationMetric):
    """
    Callable class that computes the sensitivity -> TP/(TP + FN)
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="Sensitivity", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the sensitivity score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We compute TP/(TP + FN)
        return (conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[1, 0])).item()


class Specificity(BinaryClassificationMetric):
    """
    Callable class that computes the specificity -> TN/(FP + TN)
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="Specificity", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the specificity score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """
        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We compute TN/(TN + FP)
        return (conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])).item()


class Precision(BinaryClassificationMetric):
    """
    Callable class that computes the sensitivity -> TP/(TP + FP)
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="Precision", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the precision score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We compute TP/(TP + FP)
        return (conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[0, 1])).item()


class NegativePredictiveValue(BinaryClassificationMetric):
    """
    Callable class that computes the sensitivity -> TN/(TN + FN)
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="NPV", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the precision score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We compute TN/(TN + FN)
        return (conf_mat[0, 0]/(conf_mat[0, 0] + conf_mat[1, 0])).item()


class F2Score(BinaryClassificationMetric):
    """
    Callable class that computes the ratio between binary balanced accuracy and binary cross entropy
    """
    def __init__(self,
                 n_digits: int = 5):
        """
         Builds two metric and sets other protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        self.precision = Precision(n_digits=10)
        self.sensitivity = Sensitivity(n_digits=10)
        super().__init__(direction=Direction.MAXIMIZE, name='F2_score', n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the F2-Score = (5 * Precision * Sensitivity) / (4 * Precision + Recall)

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        try:
            return (5 * self.precision(pred, targets, thresh) * self.sensitivity(pred, targets, thresh)) / \
                   (4 * self.precision(pred, targets, thresh) + self.sensitivity(pred, targets, thresh))
        except ZeroDivisionError:
            return 0.


class F1Score(BinaryClassificationMetric):
    """
    Callable class that computes the ratio between binary balanced accuracy and binary cross entropy
    """
    def __init__(self,
                 n_digits: int = 5):
        """
         Builds two metric and sets other protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        self.precision = Precision(n_digits=10)
        self.sensitivity = Sensitivity(n_digits=10)
        super().__init__(direction=Direction.MAXIMIZE, name='F1_score', n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the F1-Score = (2 * Precision * Sensitivity) / (Precision + Recall)

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """
        return (2 * self.precision(pred, targets, thresh) * self.sensitivity(pred, targets, thresh)) / \
               (self.precision(pred, targets, thresh) + self.sensitivity(pred, targets, thresh))


class NTP(BinaryClassificationMetric):
    """
    Callable class that computes the number of True Positives
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="TruePositives", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the precision score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)
        # We compute TP/(TP + FP)
        return (conf_mat[1, 1]).item()


class NFP(BinaryClassificationMetric):
    """
    Callable class that computes the number of False Positives
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="FalsePositives", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the precision score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)
        # We compute TP/(TP + FP)
        return (conf_mat[0, 1]).item()


class NFN(BinaryClassificationMetric):
    """
    Callable class that computes the number of False Positives
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="FalseNegatives", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the precision score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)
        # We compute TP/(TP + FP)
        return (conf_mat[1, 0]).item()


class NTN(BinaryClassificationMetric):
    """
    Callable class that computes the number of False Positives
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets the protected attribute of the object using parent's constructor

        Args:
            n_digits: n_digits: number of digits kept for the score
        """
        super().__init__(direction=Direction.MAXIMIZE, name="TrueNegatives", n_digits=n_digits)

    def compute_metric(self,
                       pred: tensor,
                       targets: tensor,
                       thresh: float) -> float:
        """
        Computes the precision score

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth
            thresh: probability threshold that must be reach by a sample to be classified into class 1

        Returns: float
        """

        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We compute TP/(TP + FP)
        return (conf_mat[0, 0]).item()