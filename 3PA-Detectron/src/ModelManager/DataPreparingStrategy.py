from typing import List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

class DataPreparingStrategy:
    """
    Abstract base class for data preparation strategies. 
    Subclasses should implement the `prepare_data` method, respecting the Strategy Design Pattern
    """
    @staticmethod
    def execute(features, labels=None):
        """
        Prepares data for model training or prediction.

        :param features: Features array.
        :type features: [np.ndarray, pd.DataFrame, sp.spmatrix, list]
        :param labels: Labels array, optional.
        :type labels: np.ndarray

        :return: Prepared data in the required format for the model.
        :rtype: 
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ToDmatrixStrategy(DataPreparingStrategy):
    """
    Concrete implementation for converting data into DMatrix format.
    """
    @staticmethod
    def is_supported_data(features, labels=None) -> bool:
        """
        Check if the features and labels are of supported types for creating a DMatrix.

        :param features: Features array or supported format.
        :param labels: Labels array or supported format, optional.

        :return: True if supported, False otherwise.
        :rtype: bool
        """
        # Supported data types
        supported_types = [np.ndarray, pd.DataFrame, sp.spmatrix, list]

        # Check features
        if not any(isinstance(features, t) for t in supported_types):
            return False

        # Check labels if provided
        if labels is not None and not any(isinstance(labels, t) for t in supported_types):
            return False

        return True

    @staticmethod
    def execute(features, labels=None) -> xgb.DMatrix:
        """
        Converts features and labels into a DMatrix object, handling various input types.

        :param features: Features array or supported format.
        :param labels: Labels array or supported format, optional.

        :return: A DMatrix object containing the features and labels.
        :rtype: xgb.DMatrix
        """
        # Check if data is supported
        if not ToDmatrixStrategy.is_supported_data(features, labels):
            raise ValueError("Unsupported data type provided for creating DMatrix.")

        # Create DMatrix
        return xgb.DMatrix(data=features, label=labels)


class ToNumpyStrategy(DataPreparingStrategy):
    """
    Concrete implementation of DataPreparingStrategy for converting data into NumPy array format.
    """
    @staticmethod
    def execute(features, labels=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert features and labels into NumPy arrays.

        :param features: Features array or any format that can be converted to a NumPy array.
        :param labels: Labels array or any format that can be converted to a NumPy array, optional.
        
        :return: A tuple containing the features and labels as NumPy arrays. 
            If labels are not provided, None is returned for labels.
        :rtype: (np.ndarray,np.ndarray)
        """
        features_np = np.asarray(features)
        if labels is None:
            return features_np, None
        labels_np = np.asarray(labels)

        if features_np.size == 0:
            raise ValueError("Cannot build a NumPy array from an empty features array.")
        if labels is not None and labels_np.size == 0:
            raise ValueError("Cannot build a NumPy array from an empty labels array.")

        return features_np, labels_np


class ToDataframesStrategy(DataPreparingStrategy): 
    
    def execute(self, column_labels:List[str], features, labels=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert NumPy arrays X and Y into separate pandas DataFrames with specified column labels for X.

        :param features: the features.
        :type features: np.ndarray
        :param labels: the target values.
        :type labels: np.ndarray
        :param column_labels: The column labels for the feature set.
        :type column_labels: list[str]
        
        :return: Tuple of two pandas DataFrames (X, Y)
        :rtype: (pd.DataFrame, pd.DataFrame)
        """
        # Convert X array to DataFrame
        X_df = pd.DataFrame(features, columns=column_labels)

        # Convert Y array to DataFrame without column label
        if labels is not None:
            Y_df = pd.DataFrame(labels) 
        else:
            Y_df = None
            
        return X_df, Y_df