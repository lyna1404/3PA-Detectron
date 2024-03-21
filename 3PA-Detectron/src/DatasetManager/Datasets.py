import numpy as np
from torch.utils.data import Dataset
from DataLoadingContext import DataLoadingContext

class MaskedDataset(Dataset):
    """
    A dataset wrapper for PyTorch that supports masking of data points, useful for semi-supervised learning scenarios.
    
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
        Initializes the MaskedDataset.

        Parameters:
            features (np.ndarray): The feature vectors of the dataset.
            true_labels (np.ndarray): The true labels of the dataset.
            mask (bool, optional): Indicates if the dataset is masked. Defaults to True.
            pseudo_labels (np.ndarray, optional): The pseudo labels for masked data points. Defaults to None.
        """
        self.features = features
        self.true_labels = true_labels
        self.mask = mask
        self.pseudo_labels = pseudo_labels if pseudo_labels is not None else true_labels
        self.indices = np.arange(len(self.features))
        self.original_indices = self.indices.copy()

    def __getitem__(self, index):
        """
        Retrieves the data point and its label(s) at the given index.

        Parameters:
            index (int): The index of the data point.

        Returns:
            tuple: A tuple containing the feature vector, pseudo label, true label, and mask flag for the data point.
        """
        index = self.indices[index]
        x = self.features[index]
        y = self.true_labels[index]
        y_hat = self.pseudo_labels[index]
        return x, y_hat, y, self.mask

    def refine(self, mask: np.ndarray):
        """
        Refines the dataset by applying a mask to select specific data points.

        Parameters:
            mask (np.ndarray): A boolean array indicating which data points to keep.
        """
        self.indices = self.indices[mask]

    def original(self):
        """
        Creates a new MaskedDataset instance with the original dataset without any applied mask.

        Returns:
            MaskedDataset: A new instance of the dataset with the original data.
        """
        return MaskedDataset(self.features, self.true_labels, mask=False, pseudo_labels=self.pseudo_labels)

    def reset_index(self):
        """Resets the indices of the dataset to the original indices."""
        self.indices = self.original_indices.copy()

    def __len__(self):
        """
        Gets the number of data points in the dataset.

        Returns:
            int: The number of data points.
        """
        return len(self.indices)



class DatasetsManager:
    """
    Manages various datasets for different phases of machine learning model development.
    
    This manager is responsible for loading and holding different sets of data, including training, validation,
    reference, and testing datasets.
    """
    
    def __init__(self):
        """Initializes the DatasetsManager with empty datasets."""
        self.base_model_training_set = None
        self.base_model_validation_set = None
        self.reference_set = None
        self.testing_set = None

    def set_baseModel_training_data(self, baseModel_training_file, target_column_name):
        """
        Loads and sets the base model training dataset from a file.

        Parameters:
            baseModel_training_file (str): The file path to the training data.
            target_column_name (str): The name of the target column in the dataset.
        """
        try:
            ctx = DataLoadingContext(baseModel_training_file)
            features_np, true_labels_np = ctx.load_as_np(baseModel_training_file, target_column_name)
            self.base_model_training_set = MaskedDataset(features_np, true_labels_np)
        except ValueError as e:
            print(f"Error setting base model training set: {e}")

    def set_baseModel_validation_data(self, baseModel_validation_file, target_column_name):
        """
        Loads and sets the base model validation dataset from a file.

        Parameters:
            baseModel_validation_file (str): The file path to the validation data.
            target_column_name (str): The name of the target column in the dataset.
        """
        try:
            ctx = DataLoadingContext(baseModel_validation_file)
            features_np, true_labels_np = ctx.load_as_np(baseModel_validation_file, target_column_name)
            self.base_model_validation_set = MaskedDataset(features_np, true_labels_np)
        except ValueError as e:
            print(f"Error setting base model validation set: {e}")

    
    def set_reference_data(self, reference_file, target_column_name):
        """
        Loads and sets the reference dataset from a file.

        Parameters:
            reference_file (str): The file path to the reference data.
            target_column_name (str): The name of the target column in the dataset.
        """
        try:
            ctx = DataLoadingContext(reference_file)
            features_np, true_labels_np = ctx.load_as_np(reference_file, target_column_name)
            self.reference_set = MaskedDataset(features_np, true_labels_np)
        except ValueError as e:
            print(f"Error setting reference set: {e}")

    def set_testing_data(self, testing_file, target_column_name):
        """
        Loads and sets the testing dataset from a file.

        Parameters:
            testing_file (str): The file path to the testing data.
            target_column_name (str): The name of the target column in the dataset.
        """
        try:
            ctx = DataLoadingContext(testing_file)
            features_np, true_labels_np = ctx.load_as_np(testing_file, target_column_name)
            self.testing_set = MaskedDataset(features_np, true_labels_np)
        except ValueError as e:
            print(f"Error setting testing set: {e}")

    def get_base_model_training_data(self):
        """
        getter for the trainig dataset.
        """
        if self.base_model_training_set is not None:
            return self.base_model_training_set.features, self.base_model_training_set.true_labels
        else:
            raise ValueError("Base model training set not initialized.")

    def get_base_model_validation_data(self):
        """
        getter for the validation dataset.
        """
        if self.base_model_validation_set is not None:
            return self.base_model_validation_set.features, self.base_model_validation_set.true_labels
        else:
            raise ValueError("Base model validation set not initialized.")

    def get_reference_data(self):
        """
        getter for the reference dataset.
        """
        if self.reference_set is not None:
            return self.reference_set.features, self.reference_set.true_labels
        else:
            raise ValueError("Reference set not initialized.")

    def get_testing_data(self):
        """
        getter for the testing dataset.
        """
        if self.testing_set is not None:
            return self.testing_set.features, self.testing_set.true_labels
        else:
            raise ValueError("Testing set not initialized.")

