import numpy as np
import xgboost as xgb

class DataPreparingStrategy:
    """
    Abstract base class for data preparation strategies.
    """
    @staticmethod
    def execute(features, labels=None):
        """
        Prepares data for model training or prediction.

        :param features: Features array.
        :param labels: Labels array, optional.
        :return: Prepared data in the required format for the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class ToDmatrixStrategy(DataPreparingStrategy):
    """
    Concrete implementation of DataPreparingStrategy for converting data into DMatrix format.
    """
    @staticmethod
    def execute(features: np.ndarray, labels: np.ndarray = None):
        """
        Converts features and labels into a DMatrix object.

        :param features: Features array.
        :param labels: Labels array, optional.
        :return: A DMatrix object containing the features and labels.
        """
        if features.size == 0:
            raise ValueError("Cannot build a DMatrix from an empty features array.")
        if labels is not None and labels.size == 0:
            raise ValueError("Cannot build a DMatrix from an empty labels array.")
        
        return xgb.DMatrix(data=features, label=labels)
