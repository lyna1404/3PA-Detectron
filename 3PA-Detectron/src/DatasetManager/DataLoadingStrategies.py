import pandas as pd


class DataLoadingStrategy:
    """
    An abstract class representing the loading strategy of datasets.
    
    It provides a way to load datasets while adhering to the Strategy Design Pattern.
    """
    @staticmethod
    def execute(path_to_file:str, target_column_name:str):
        """
        A method that should be implemented by subclasses to load datasets.
        
        :param path_to_file: The path to the dataset file.
        :type path_to_file: str
        :param target_column_name: The name of the column to be used as the target variable.
        :type target_column_name: str

        :return: A tuple containing a list of column labels and NumPy arrays (data features and data targets).
        :rtype: (list, np.ndarray,np.ndarray)
        """
        pass


class CSVDataLoadingStrategy(DataLoadingStrategy):
    """
    A concrete subclass of DataLoadingStrategy that defines loading behavior for CSV tabular datasets.
    """
    @staticmethod
    def execute(path_to_file:str, target_column_name:str):
        """
        Read a CSV file and load it into a numpy arrays  

        :param path_to_file: The path to the csv file
        :type path_to_file: str
        :param target_column_name: The name of the column which will be used as target
        :type target_column_name: str

        :return: a list of the column labels and numpy arrays (data features and data targets)
        :rtype: (list, np.ndarray,np.ndarray)
        """
        # Read the CSV file
        df = pd.read_csv(path_to_file)
        
        # Separate features and target
        features = df.drop(columns=[target_column_name])  
        target = df[target_column_name]  
        column_labels = features.columns.tolist()

        # Convert to NumPy arrays
        features_np = features.to_numpy()
        target_np = target.to_numpy()
        
        return column_labels, features_np, target_np

