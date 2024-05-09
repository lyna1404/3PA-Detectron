from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    precision_score, f1_score
)
from typing import Union

class EvaluationMetric(ABC):
    """"
    Abstract base class for evaluation metrics. Subclasses should implement the `calculate` method.
    """
    @abstractmethod
    def calculate(self, y_true:np.ndarray, y_pred_score:np.ndarray, sample_weight:np.ndarray=None) ->  Union[float, np.ndarray]:
        """
        Calculate the desired evaluation metric.

        :param y_true: The true labels.
        :type y_trye: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The calculated evaluation metric.
        """
        pass


class Accuracy(EvaluationMetric):
    """
    Concrete Class to calculate the accuracy metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the accuracy metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The accuracy score.
        :rtype: float
        """
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return accuracy_score(y_true, y_pred_score, sample_weight=sample_weight)


class Recall(EvaluationMetric):
    """
    Concrete Class to calculate the recall metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> Union[float, np.ndarray]:
        """
        Calculate the accuracy metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The recall score.
        :rtype: float | np.ndarray
        """
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return recall_score(y_true, y_pred_score, sample_weight=sample_weight, zero_division=0)


class RocAuc(EvaluationMetric):
    """
    Concrete Class to calculate the AUC metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the AUC metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The AUC score.
        :rtype: float 
        """
        if len(np.unique(y_true)) == 1:
            return np.nan
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return roc_auc_score(y_true, y_pred_score, sample_weight=sample_weight)


class AveragePrecision(EvaluationMetric):
    """
    Concrete Class to calculate the Average Precision metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the Average Precision metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The Average Precision score.
        :rtype: float 
        """
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return average_precision_score(y_true, y_pred_score, sample_weight=sample_weight)


class MatthewsCorrCoef(EvaluationMetric):
    """
    Concrete Class to calculate the Matthews Correlation Coefficient metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the Matthews Correlation Coefficient metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The Matthews Correlation Coefficient score.
        :rtype: float 
        """
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return matthews_corrcoef(y_true, y_pred_score, sample_weight=sample_weight)


class Precision(EvaluationMetric):
    """
    Concrete Class to calculate the Precision metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> Union[float, np.ndarray]:
        """
        Calculate the Precision metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The Precision score.
        :rtype: float | np.ndarray 
        """
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return precision_score(y_true, y_pred_score, sample_weight=sample_weight, zero_division=0)


class F1Score(EvaluationMetric):
    """
    Concrete Class to calculate the F1Score metric.
    """
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None) -> Union[float, np.ndarray]:
        """
        Calculate the F1Score metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The F1Score.
        :rtype: float | np.ndarray 
        """
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return f1_score(y_true, y_pred_score, sample_weight=sample_weight, zero_division=0)


class Sensitivity(EvaluationMetric):
    """
    Concrete Class to calculate the Sensitivity metric.
    """
    def calculate(self, y_true:np.ndarray, y_pred_score:np.ndarray, sample_weight:np.ndarray=None) -> Union[float, np.ndarray]:
        """
        Calculate the Sensitivity metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The Sensitivity result.
        :rtype: float | np.ndarray 
        """      
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        return recall_score(y_true, y_pred_score, pos_label=1, zero_division=0)


class Specificity(EvaluationMetric):
    """
    Concrete Class to calculate the Specificity metric.
    """
    def calculate(self, y_true:np.ndarray, y_pred_score:np.ndarray, sample_weight:np.ndarray=None) -> Union[float, np.ndarray]:
        """
        Calculate the Specificity metric.
        For specificity, we need to reverse the true labels and predictions of the model

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The Specificity result.
        :rtype: float | np.ndarray 
        """      
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        # Reverse True Labels and Predictions
        return recall_score(y_true, y_pred_score, pos_label=0, zero_division=0)


class PPV(EvaluationMetric):
    """
    Concrete Class to calculate the Positive predictive value.
    """
    def calculate(self, y_true:np.ndarray, y_pred_score:np.ndarray, sample_weight:np.ndarray=None) -> Union[float, np.ndarray]:
        """
        Calculate the Positive predictive value.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The PPV result.
        :rtype: float | np.ndarray 
        """  
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        return precision_score(y_true, y_pred_score, pos_label=1, zero_division=0)


class NPV(EvaluationMetric):
    """
    Concrete Class to calculate the Negative predictive value.
    """
    def calculate(self, y_true:np.ndarray, y_pred_score:np.ndarray, sample_weight:np.ndarray=None) -> Union[float, np.ndarray]:
        """
        Calculate the Negative predictive value.
        Akin to the Specificity, for NPV, we need to reverse the labels and predictions

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The NPV result.
        :rtype: float | np.ndarray 
        """  
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        # Reverse the True labels and predictions
        return precision_score(y_true, y_pred_score, pos_label=0, zero_division=0)


class BalancedAccuracy(EvaluationMetric):
    """
    Concrete Class to calculate the Balanced Accuracy.
    """
    def calculate(self, y_true:np.ndarray, y_pred_score:np.ndarray, sample_weight:np.ndarray=None) -> float:
        """
        Calculate the Balanced Accuracy metric.

        :param y_true: The true labels.
        :type y_true: np.ndarray
        :param y_pred_score: The predicted scores.
        :type y_pred_score: np.ndarray
        :param sample_weight: Sample weights (optional).
        :type sample_weight: np.ndarray

        :return: The Balanced Accuracy result.
        :rtype: float | np.ndarray 
        """  
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        sens = Sensitivity().calculate(y_true, y_pred_score)
        spec = Specificity().calculate(y_true, y_pred_score)
        return (sens + spec) / 2