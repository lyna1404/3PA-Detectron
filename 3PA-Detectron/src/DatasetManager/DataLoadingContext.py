import DataLoadingStrategies

class DataLoadingContext:
    """
    Context class for loading datasets using different loading strategies.
    
    Attributes:
    file path (str): a string containing the file path to the dataset
    """
    __strategies = {
        'csv': DataLoadingStrategies.CSVDataLoadingStrategy,
    }
    
    def __init__(self, file_path:str):
        """
        Initialize the context with the appropriate loading strategy based on the file extension.

        :param file_path: The path to the dataset file.
        :type file_path: str
        """
        file_extension = file_path.split('.')[-1] 
        strategy_class = self.__strategies.get(file_extension, None)
        if strategy_class is None:
            raise ValueError(f"This file extension is not supported yet ! '{file_extension}'")
        self.selected_strategy = strategy_class()
    
    def set_strategy(self, strategy: DataLoadingStrategies.DataLoadingStrategy):
        """
        Set the loading strategy to be used.

        :param strategy: The loading strategy to be set.
        :type strategy: DataLoadingStrategies.DataLoadingStrategy
        """
        self.selected_strategy = strategy
    
    def get_strategy(self):
        """
        Get the currently selected loading strategy.

        :return: The currently selected loading strategy.
        :rtype: DataLoadingStrategy
        """
        return self.selected_strategy
    
    def load_as_np(self, file_path:str, target_column_name:str):
        """
        Load the dataset as NumPy arrays using the selected loading strategy.

        :param file_path: The path to the dataset file.
        :type file_path: str
        :param target_column_name: The name of the column to be used as the target variable.
        :type target_column_name: str
        
        :return: A list of the column labels and numpy arrays (data features and data targets)
        :rtype: (list[str], np.ndarray, np.ndarray )
        """
        return self.selected_strategy.execute(file_path, target_column_name)
