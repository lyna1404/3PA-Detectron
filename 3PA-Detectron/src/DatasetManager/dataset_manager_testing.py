import numpy as np
import os
import unittest

from DataLoadingContext import DataLoadingContext
from DataLoadingStrategies import CSVDataLoadingStrategy
from Datasets import DatasetsManager


class TestDatasetsManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a simple CSV for testing."""
        cls.sample_csv = 'sample_data.csv'
        cls.target_column = 'target'
        with open(cls.sample_csv, 'w') as f:
            f.write("feature1,feature2,target\n1,2,0\n3,4,1")  # Example data

    def test_CSVDataLoadingStrategy(self):
        """Test loading data from a CSV file using the CSVDataLoadingStrategy."""
        features, target = CSVDataLoadingStrategy.execute(self.sample_csv, self.target_column)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertEqual(features.shape[0], target.shape[0])

    def test_DataLoadingContext_with_CSV(self):
        """Test the DataLoadingContext's ability to use the correct strategy for CSV files."""
        ctx = DataLoadingContext(self.sample_csv)
        strategy = ctx.get_strategy()
        self.assertIsInstance(strategy, CSVDataLoadingStrategy)

    def test_DatasetsManager_baseModel_training_data_loading(self):
        """Test DatasetsManager's ability to load and retrieve base model training data."""
        manager = DatasetsManager()
        manager.set_baseModel_training_data(self.sample_csv, self.target_column)
        features, labels = manager.get_base_model_training_data()
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(features), len(labels))
   
    def test_DatasetsManager_baseModel_testing_data_loading(self):
        """Test DatasetsManager's ability to load and retrieve testing data."""
        manager = DatasetsManager()
        manager.set_testing_data(self.sample_csv, self.target_column)
        features, labels = manager.get_testing_data()
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(features), len(labels))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        os.remove(cls.sample_csv)


if __name__ == '__main__':
    unittest.main()
