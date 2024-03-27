from abc import ABC, abstractmethod
import Numpy as np

class UncertaintyMetric(ABC):
    @abstractmethod
    def calculate(x, y_pred, y_true):
        pass

class AbsoluteError(UncertaintyMetric):
    def calculate(x : np.ndarray, predicted_prob : np.ndarray, y_true : np.ndarray):
        return 1 - np.abs(y_true - predicted_prob)
    
class UncertaintyCalculator:
    def __init__(self, metric : UncertaintyMetric) -> None:
        self.metric = metric
    
    def calculate_uncertainty(self, x : np.ndarray, predicted_prob : np.ndarray, y_true : np.ndarray):
        return self.metric.calculate(x, predicted_prob, y_true)