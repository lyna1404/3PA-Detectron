import pandas as pd
import numpy as np

class DataLoadingStrategy:
    @staticmethod
    def execute(path_to_file, target_column_name):
        pass

class CSVDataLoadingStrategy(DataLoadingStrategy):
    @staticmethod
    def execute(path_to_file, target_column_name):
        # Read the CSV file
        df = pd.read_csv(path_to_file)
        
        # Separate features and target
        features = df.drop(columns=[target_column_name])  
        target = df[target_column_name]  
        
        # Convert to NumPy arrays
        features_np = features.to_numpy()
        target_np = target.to_numpy()
        
        return features_np, target_np

