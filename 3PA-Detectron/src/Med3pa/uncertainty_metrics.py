from abc import ABC, abstractmethod
import numpy as np

class UncertaintyMetric(ABC):
    """
    Abstract base class for calculating uncertainty metrics.
    """
    @abstractmethod
    def calculate(x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate a general uncertainty metric.

        :param x: The input data.
        :type x: np.ndarray
        :param y_pred: The predicted probabilities.
        :type y_pred: np.ndarray
        :param y_true: The true labels.
        :type y_true: np.ndarray

        :return: The calculated uncertainty metric.
        :rtype: np.ndarray
        """
        pass


class AbsoluteError(UncertaintyMetric):
    """
    Calculates the absolute error uncertainty metric.
    """
    def calculate(x : np.ndarray, predicted_prob : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        """
        Calculate the absolute error uncertainty metric.

        :param x: The input data.
        :type x: np.ndarray
        :param predicted_prob: The predicted probabilities.
        :type predicted_prob: np.ndarray
        :param y_true: The true labels.
        :type y_true: np.ndarray

        :return: The calculated absolute error uncertainty metric.
        :rtype: np.ndarray
        """
        return 1 - np.abs(y_true - predicted_prob)


class UncertaintyCalculator:
    """
    Calculates uncertainty based on a specified metric.

    Attributes: 
        metric (UncertaintyMetric): The specified Uncertainty metric to use for calculation.
    """
    def __init__(self, metric : UncertaintyMetric) -> None:
        """
        Initialize the UncertaintyCalculator.

        Specify the uncertainty metric to use for calculating model uncertainty.

        :param metric: The uncertainty metric to use.
        :type metric: UncertaintyMetric
        """
        self.metric = metric
    
    def calculate_uncertainty(self, x : np.ndarray, predicted_prob : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        """
        Calculate the uncertainty using the specified metric.

        :param x: The input data.
        :type x: np.ndarray
        :param predicted_prob: The predicted probabilities.
        :type predicted_prob: np.ndarray
        :param y_true: The true labels.
        :type y_true: np.ndarray
        
        :return: The calculated uncertainty.
        :rtype: np.ndarray
        """
        return self.metric.calculate(x, predicted_prob, y_true)