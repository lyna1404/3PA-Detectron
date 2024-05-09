import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

class DataPreparingStrategy:
    """
    Abstract base class for data preparation strategies.
    """
    @staticmethod
    def execute(features, labels=None, weights=None):
        """
        Prepares data for model training or prediction.

        :param features: Features array.
        :param labels: Labels array, optional.
        :return: Prepared data in the required format for the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class ToDmatrixStrategy(DataPreparingStrategy):
    """
    Concrete implementation for converting data into DMatrix format.
    """
    @staticmethod
    def is_supported_data(features, labels=None, weights=None):
        """
        Check if the features and labels are of supported types for creating a DMatrix.

        :param features: Features array or supported format.
        :param labels: Labels array or supported format, optional.
        :return: True if supported, False otherwise.
        """
        # Supported data types
        supported_types = [np.ndarray, pd.DataFrame, sp.spmatrix, list]

        # Check features
        if not any(isinstance(features, t) for t in supported_types):
            return False
        
        # Check weights if provided
        if weights is not None and not any(isinstance(weights, t) for t in supported_types):
            return False

        # Check labels if provided
        if labels is not None and not any(isinstance(labels, t) for t in supported_types):
            return False

        return True

    @staticmethod
    def execute(features, labels=None, weights =None):
        """
        Converts features and labels into a DMatrix object, handling various input types.

        :param features: Features array or supported format.
        :param labels: Labels array or supported format, optional.
        :return: A DMatrix object containing the features and labels.
        """
        # Check if data is supported
        if not ToDmatrixStrategy.is_supported_data(features, labels):
            raise ValueError("Unsupported data type provided for creating DMatrix.")

        # Create DMatrix
        return xgb.DMatrix(data=features, label=labels, weight=weights)

class ToNumpyStrategy(DataPreparingStrategy):
    """
    Concrete implementation of DataPreparingStrategy for converting data into NumPy array format.
    """
    @staticmethod
    def execute(features, labels=None, weights=None):
        """
        Converts features and labels into NumPy arrays.

        :param features: Features array or any format that can be converted to a NumPy array.
        :param labels: Labels array or any format that can be converted to a NumPy array, optional.
        :return: A tuple containing the features and labels as NumPy arrays. If labels are not provided, None is returned for labels.
        """
        features_np = np.asarray(features)
        if labels is None:
            return features_np, None
        labels_np = np.asarray(labels)
        weights_np = np.asarray(weights) if weights is not None else None

        if features_np.size == 0:
            raise ValueError("Cannot build a NumPy array from an empty features array.")
        if labels is not None and labels_np.size == 0:
            raise ValueError("Cannot build a NumPy array from an empty labels array.")

        return features_np, labels_np, weights_np
    
class ToDataframesStrategy(DataPreparingStrategy): 
    
    def execute(self, column_labels, features, labels=None, weights=None):
        """
        Convert NumPy arrays X and Y into separate pandas DataFrames with specified column labels for X.

        Parameters:
        - features: NumPy array representing the features.
        - labels: NumPy array representing the target values.
        - column_labels: List containing column labels for X.

        Returns:
        - X_df: A pandas DataFrame with X and its corresponding column labels.
        - Y_df: A pandas DataFrame with Y.
        """
        # Convert X array to DataFrame
        X_df = pd.DataFrame(features, columns=column_labels)

        # Convert Y array to DataFrame without column label
        if labels is not None:
            Y_df = pd.DataFrame(labels) 
        else:
            Y_df = None
        
        W_df = pd.DataFrame(weights) if weights is not None else None

        return X_df, Y_df, W_df