from .DataLoadingStrategies import DataLoadingStrategy, CSVDataLoadingStrategy

class DataLoadingContext:
    __strategies = {
        'csv': CSVDataLoadingStrategy,
    }
    def __init__(self, file_path):
        file_extension = file_path.split('.')[-1] 
        strategy_class = self.__strategies.get(file_extension, None)
        if strategy_class is None:
            raise ValueError(f"This file extension is not supported yet ! '{file_extension}'")
        self.selected_strategy = strategy_class()
    
    def set_strategy(self, strategy : DataLoadingStrategy) :
        self.selected_strategy = strategy
    
    def get_strategy(self):
        return self.selected_strategy
    
    def load_as_np(self, file_path, target_column_name):
        return self.selected_strategy.execute(file_path, target_column_name)
