import numpy as np
from torch.utils.data import Dataset
from .DataLoadingContext import DataLoadingContext

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
    
    def __init__(self, features: np.ndarray, true_labels: np.ndarray, mask=True, pseudo_labels: np.ndarray = None, pseudo_probabilities = None):
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
        self.pseudo_labels = pseudo_labels 
        self.pseudo_probabilities = pseudo_probabilities 
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
        return x, y_hat, y

    def refine(self, mask: np.ndarray):
        """
        Refines the dataset by applying a mask to select specific data points.

        Parameters:
            mask (np.ndarray): A boolean array indicating which data points to keep.
        """
        self.indices = self.indices[mask]
        self.features = self.features[mask]
        self.true_labels = self.true_labels[mask]
        self.pseudo_labels = self.pseudo_labels[mask]
        self.pseudo_probabilities = self.pseudo_probabilities[mask]
        return len(self.pseudo_labels)
    
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

    def sample(self, N: int, seed:int):
        """
        Samples N data points from the dataset and returns a new MaskedDataset instance containing these samples.

        Parameters:
            N (int): The number of samples to return.

        Returns:
            MaskedDataset: A new instance of the dataset containing N random samples.
        """
        if N > len(self.indices):
            raise ValueError("N cannot be greater than the current number of data points in the dataset.")
        
        sampled_indices = np.random.RandomState(seed=seed).permutation(len(self.features))[:(N := N)]
        sampled_features = self.features[sampled_indices, :]
        sampled_true_labels = self.true_labels[sampled_indices]
        sampled_pseudo_labels = self.pseudo_labels[sampled_indices] if self.pseudo_labels is not None else None
        
        return MaskedDataset(sampled_features, sampled_true_labels, sampled_pseudo_labels)

    def get_features(self):
        return self.features
    
    def get_pseudo_labels(self):
        return self.pseudo_labels
    
    def get_true_labels(self):
        return self.true_labels
    
    def get_pseudo_probabilities(self):
        return self.pseudo_probabilities
    
    def set_pseudo_probs_labels(self, pseudo_probabilities: np.ndarray, threshold=0.5):
        """
        Sets the pseudo probabilities and corresponding pseudo labels for the dataset. The labels are derived by
        applying a threshold to the probabilities.

        Parameters:
            pseudo_probabilities (np.ndarray): The pseudo probabilities array to be set. This can be a vector of
                                            probabilities (for binary classification) or a matrix where each row
                                            corresponds to the probability distribution over classes for each sample
                                            (for multiclass classification).
            threshold (float, optional): The threshold to convert probabilities to binary labels. Defaults to 0.5.

        Raises:
            ValueError: If the shape of pseudo_probabilities does not match the number of samples in the features array.
        """
        # Check if the number of pseudo probabilities matches the number of features
        if pseudo_probabilities.shape[0] != self.features.shape[0]:
            raise ValueError("The shape of pseudo_probabilities must match the number of samples in the features array.")
        
        # Set pseudo probabilities
        self.pseudo_probabilities = pseudo_probabilities
        self.pseudo_labels = pseudo_probabilities > threshold
        
    def clone(self):
        """
        Creates a clone of the current MaskedDataset instance.

        Returns:
            MaskedDataset: A new instance of MaskedDataset containing the same data and configurations as the current instance.
        """
        # Create a new instance of MaskedDataset with the same data and attributes
        cloned_dataset = MaskedDataset(
            features=self.features.copy(),
            true_labels=self.true_labels.copy(),
            pseudo_labels=self.pseudo_labels.copy() if self.pseudo_labels is not None else None,
            pseudo_probabilities=self.pseudo_probabilities.copy() if self.pseudo_probabilities is not None else None
        )
        cloned_dataset.indices = self.indices.copy()
        cloned_dataset.original_indices = self.original_indices.copy()
        return cloned_dataset

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
        self.column_labels = None

    def set_baseModel_training_data(self, baseModel_training_file, target_column_name):
        """
        Loads and sets the base model training dataset from a file.

        Parameters:
            baseModel_training_file (str): The file path to the training data.
            target_column_name (str): The name of the target column in the dataset.
        """
        try:
            ctx = DataLoadingContext(baseModel_training_file)
            column_labels, features_np, true_labels_np = ctx.load_as_np(baseModel_training_file, target_column_name)
            self.base_model_training_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the training dataset do not match "
                                    "the existing column labels.")
                
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
            column_labels, features_np, true_labels_np = ctx.load_as_np(baseModel_validation_file, target_column_name)
            self.base_model_validation_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the validation dataset do not match "
                                    "the existing column labels.")
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
            column_labels, features_np, true_labels_np = ctx.load_as_np(reference_file, target_column_name)
            self.reference_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the reference dataset do not match "
                                    "the existing column labels.")
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
            column_labels, features_np, true_labels_np = ctx.load_as_np(testing_file, target_column_name)
            self.testing_set = MaskedDataset(features_np, true_labels_np)
            # If self.column_labels is None, set it
            if self.column_labels is None:
                self.column_labels = column_labels
            else:
                # Compare extracted column labels with self.column_labels
                if self.column_labels != column_labels:
                    raise ValueError("Column labels extracted from the testing dataset do not match "
                                    "the existing column labels.")
        except ValueError as e:
            print(f"Error setting testing set: {e}")

    def get_base_model_training_data(self, return_instance: bool = False):
        """
        getter for the trainig dataset.
        """
        if self.base_model_training_set is not None :
            if not return_instance:
                return self.base_model_training_set.features, self.base_model_training_set.true_labels
            else:
                return self.base_model_training_set
        else:
            raise ValueError("Base model training set not initialized.")

    def get_base_model_validation_data(self, return_instance: bool = False):
        """
        getter for the validation dataset.
        """
        if self.base_model_validation_set is not None:
            if not return_instance:
                return self.base_model_validation_set.features, self.base_model_validation_set.true_labels
            else:
                return self.base_model_validation_set
        else:
            raise ValueError("Base model validation set not initialized.")

    def get_reference_data(self, return_instance: bool = False):
        """
        getter for the reference dataset.
        """
        if self.reference_set is not None:
            if not return_instance:
                return self.reference_set.features, self.reference_set.true_labels
            else:
                return self.reference_set
        else:
            raise ValueError("Reference set not initialized.")

    def get_testing_data(self, return_instance: bool = False):
        """
        getter for the testing dataset.
        """
        if self.testing_set is not None:
            if not return_instance:
                return self.testing_set.features, self.testing_set.true_labels
            else:
                return self.testing_set
        else:
            raise ValueError("Testing set not initialized.")
