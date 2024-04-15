from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score, matthews_corrcoef, \
    precision_score, f1_score

class EvaluationMetric(ABC):
    @abstractmethod
    def calculate(self, y_true, y_pred_score, sample_weight=None):
        pass

class Accuracy(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return accuracy_score(y_true, y_pred_score, sample_weight=sample_weight)

class Recall(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return recall_score(y_true, y_pred_score, sample_weight=sample_weight, zero_division=0)

class RocAuc(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if len(np.unique(y_true)) == 1:
            return np.nan
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return roc_auc_score(y_true, y_pred_score, sample_weight=sample_weight)

class AveragePrecision(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return average_precision_score(y_true, y_pred_score, sample_weight=sample_weight)

class MatthewsCorrCoef(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return matthews_corrcoef(y_true, y_pred_score, sample_weight=sample_weight)

class Precision(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return precision_score(y_true, y_pred_score, sample_weight=sample_weight, zero_division=0)

class F1Score(EvaluationMetric):
    def calculate(self, y_true: np.ndarray, y_pred_score: np.ndarray, sample_weight: np.ndarray = None):
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return f1_score(y_true, y_pred_score, sample_weight=sample_weight, zero_division=0)
    
class Sensitivity(EvaluationMetric):
    def calculate(self, y_true, y_pred_score, sample_weight=None):
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        return recall_score(y_true, y_pred_score, pos_label=1, zero_division=0)

class Specificity(EvaluationMetric):
    def calculate(self, y_true, y_pred_score, sample_weight=None):
        # For specificity, we need to reverse the labels and predictions
        if y_true.size == 0 or y_pred_score.size == 0:
                    return np.nan
        return recall_score(y_true, y_pred_score, pos_label=0, zero_division=0)

class PPV(EvaluationMetric):
    def calculate(self, y_true, y_pred_score, sample_weight=None):
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        return precision_score(y_true, y_pred_score, pos_label=1, zero_division=0)

class NPV(EvaluationMetric):
    def calculate(self, y_true, y_pred_score, sample_weight=None):
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        # For NPV, we need to reverse the labels and predictions
        return precision_score(y_true, y_pred_score, pos_label=0, zero_division=0)

class BalancedAccuracy(EvaluationMetric):
    def calculate(self, y_true, y_pred_score, sample_weight=None):
        if y_true.size == 0 or y_pred_score.size == 0:
            return np.nan
        sens = Sensitivity().calculate(y_true, y_pred_score)
        spec = Specificity().calculate(y_true, y_pred_score)
        return (sens + spec) / 2