import numpy as np
from torch.utils.data import Dataset
from DataLoadingContext import DataLoadingContext


class MaskedDataset(Dataset):
    """
    A dataset wrapper for PyTorch that supports masking of data points, 
    useful for semi-supervised learning scenarios.
    
    Attributes:
        features (np.ndarray): The feature vectors of the dataset.
        true_labels (np.ndarray): The true labels of the dataset.
        mask (bool): A flag indicating whether masking is applied.
        pseudo_labels (np.ndarray): The pseudo labels used when mask is True.
        indices (np.ndarray): The current indices of the dataset after applying masking.
        original_indices (np.ndarray): The original indices of the dataset.
    """
    
    def __init__(self, features: np.ndarray, true_labels: np.ndarray, mask=True, pseudo_labels: np.ndarray = None):
        """
        Initialize the MaskedDataset.

        
        :param features: The feature vectors of the dataset.
        :type features:  np.ndarray
        :param true_labels: The true labels of the dataset.
        :type true_labels: np.ndarray
        :param mask: Indicates if the dataset is masked. Defaults to True.
        :type mask: (bool, optional)
        :param pseudo_labels: The pseudo labels for masked data points. Defaults to None.
        :type pseudo_labels: (np.ndarray, optional)
        """
        self.features = features
        self.true_labels = true_labels
        self.mask = mask
        self.pseudo_labels = pseudo_labels if pseudo_labels is not None else true_labels
        self.indices = np.arange(len(self.features))
        self.original_indices = self.indices.copy()

    def __getitem__(self, index:int):
        """
        Retrieve the data point and its label(s) at the given index.

        :param index: The index of the data point.
        :type index: int

        :return: A tuple containing the feature vector, the true, pseudo label and the
          mask flag for the data point.
        :rtype: (np.ndarray,np.ndarray,np.ndarray)
        """
        index = self.indices[index]
        x = self.features[index]
        y = self.true_labels[index]
        y_hat = self.pseudo_labels[index]
        return x, y_hat, y, self.mask

    def refine(self, mask: np.ndarray):
        """
        Refine the dataset by applying a mask to select specific data points.

        :param mask: A boolean array indicating which data points to keep.
        :type mask: np.ndarray
        """
        self.indices = self.indices[mask]

    def original(self):
        """
        Create a new MaskedDataset instance with the original dataset without any applied mask.

        :param MaskedDataset: A new instance of the dataset with the original data. 
        :type MaskedDataset: MaskedDataset
        """
        return MaskedDataset(self.features, self.true_labels, mask=False, pseudo_labels=self.pseudo_labels)

    def reset_index(self):
        """Reset the indices of the dataset to the original indices."""
        self.indices = self.original_indices.copy()

    def __len__(self):
        """
        Get the number of data points in the dataset.

        
        :return: The number of data points.
        :rtype: int
        """
        return len(self.indices)


class DatasetsManager:
    """
    Manage various datasets for different phases of machine learning model development.
    
    This manager is responsible for loading and holding different sets of data, 
    including training, validation,reference, and testing datasets.
    """
    
    def __init__(self):
        """
        Initialize the DatasetsManager with empty datasets.
        """
        self.base_model_training_set = None
        self.base_model_validation_set = None
        self.reference_set = None
        self.testing_set = None
        self.column_labels = None

    def set_baseModel_training_data(self, baseModel_training_file:str, target_column_name:str):
        """
        Load and set the base model training dataset from a file.

        :param baseModel_training_file: The file path to the training data.
        :type baseModel_training_file: str
        :param target_column_name: The name of the column that contains the target variable.
        :type target_column_name: str

        :raises ValueError: Raises an assertion error if the name of columns in the training data 
        does not match the given column names.
        :raises ValueError: If loading the training set fails.
        """
        try:
            ctx = DataLoadingContext(baseModel_training_file)
            column_labels, features_np, true_labels_np = ctx.load_as_np(baseModel_training_file, 
                                                                        target_column_name)
            self.base_model_training_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the dataset do not match "
                                    "the existing column labels.")
                
        except ValueError as e:
            print(f"Error setting base model training set: {e}")

    def set_baseModel_validation_data(self, baseModel_validation_file:str, target_column_name:str):
        """
        Loas and set the base model validation dataset from a file.

        :param baseModel_validation_file: The file path to the validation data.
        :type baseModel_validation_file: str
        :param target_column_name: The name of the target column in the dataset.
        :type target_column_name: str

        :raises ValueError: Raises an assertion error if the name of columns in the validation data 
        does not match the given column names.
        :raises ValueError: If loading the validation set fails.
        """
        try:
            ctx = DataLoadingContext(baseModel_validation_file)
            column_labels, features_np, true_labels_np = ctx.load_as_np(baseModel_validation_file, 
                                                                        target_column_name)
            self.base_model_validation_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the dataset do not match "
                                    "the existing column labels.")
        except ValueError as e:
            print(f"Error setting base model validation set: {e}")

    
    def set_reference_data(self, reference_file:str, target_column_name:str):
        """
        Load and set the reference dataset from a file.

        :param reference_file: The file path to the reference data.
        :type reference_file: str
        :param target_column_name: The name of the target column in the dataset.
        :type target_column_name: str

        :raises ValueError: Raises an assertion error if the name of columns in the reference data 
        does not match the given column names.
        :raises ValueError: If loading the reference dataset fails.
        """
        try:
            ctx = DataLoadingContext(reference_file)
            column_labels, features_np, true_labels_np = ctx.load_as_np(reference_file, 
                                                                        target_column_name)
            self.reference_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the dataset do not match "
                                    "the existing column labels.")
        except ValueError as e:
            print(f"Error setting reference set: {e}")

    def set_testing_data(self, testing_file:str, target_column_name:str):
        """
        Load and set the testing dataset from a file.

        :param testing_file: The file path to the testing data.
        :type testing_file: str
        :param target_column_name: The name of the target column in the dataset.
        :type target_column_name: str

        :raises ValueError: Raises an assertion error if the name of columns in the testing data 
        does not match the given column names.
        :raises ValueError: If loading the testing set fails.
        """
        try:
            ctx = DataLoadingContext(testing_file)
            features_np, true_labels_np = ctx.load_as_np(testing_file, target_column_name)
            column_labels, self.testing_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the dataset do not match "
                                    "the existing column labels.")
        except ValueError as e:
            print(f"Error setting testing set: {e}")

    def get_base_model_training_data(self):
        """
        getter for the trainig dataset.
        
        :return: A tuple containing the feature matrix X and the label vector y.
        :rtype: (np.ndarray,np.ndarray)

        :raises ValueError: If the Base model training set is not initialized
        """
        if self.base_model_training_set is not None:
            return self.base_model_training_set.features, self.base_model_training_set.true_labels
        else:
            raise ValueError("Base model training set not initialized.")

    def get_base_model_validation_data(self):
        """
        getter for the validation dataset.

        :return: A tuple containing the feature matrix X and the label vector y.
        :rtype: (np.ndarray,np.ndarray)

        :raises ValueError: If the Base model validation set is not initialized
        """
        if self.base_model_validation_set is not None:
            return self.base_model_validation_set.features, self.base_model_validation_set.true_labels
        else:
            raise ValueError("Base model validation set not initialized.")

    def get_reference_data(self):
        """
        getter for the reference dataset.

        :return: A tuple containing the feature matrix X and the label vector y.
        :rtype: (np.ndarray,np.ndarray)

        :raises ValueError: If the Base model reference set is not initialized
        """
        if self.reference_set is not None:
            return self.reference_set.features, self.reference_set.true_labels
        else:
            raise ValueError("Reference set not initialized.")

    def get_testing_data(self):
        """
        getter for the testing dataset.

        :return: A tuple containing the feature matrix X and the label vector y.
        :rtype: (np.ndarray,np.ndarray)

        :raises ValueError: If the Base model testing set is not initialized
        """
        if self.testing_set is not None:
            return self.testing_set.features, self.testing_set.true_labels
        else:
            raise ValueError("Testing set not initialized.")

