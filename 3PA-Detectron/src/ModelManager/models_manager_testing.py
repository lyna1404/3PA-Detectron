import unittest
import numpy as np
import xgboost as xgb
from ModelFactories import ModelFactory, XGBoostFactory
from Models import XGBoostModel
from DataPreparingStrategy import ToDmatrixStrategy, DataPreparingStrategy
from BaseModel import BaseModelManager
from xgboost.sklearn import XGBClassifier

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.features = np.array([[1, 2], [4, 6]])
        self.labels = np.array([1, 0])
        self.hyperparams = {'objective': 'binary:logistic'}
    def test_factory_and_model_integration_using_params(self):
        """Test model creation from params using the factory and its functionality."""
        model = ModelFactory.create_model_with_hyperparams('XGBoostModel', self.hyperparams)
        self.assertIsInstance(model, XGBoostModel)
        # Basic training and prediction test
        model.train(self.features, self.labels)
        predictions = model.predict(self.features)
        self.assertEqual(len(predictions), len(self.labels))
    
    def test_model_from_pickled_file_booster(self):
        """Test model loading from a pickled file and its functionality."""
        # Assume 'xgboost_model.pkl' is the pickled model file created earlier
        pickled_model_path = 'booster_model.pkl'
        loaded_model = ModelFactory.create_model_from_pickled(pickled_model_path)
        self.assertIsInstance(loaded_model, XGBoostModel)
        self.assertIsInstance(loaded_model.model, xgb.Booster)

        # Verify that the loaded model can make predictions (assumes model was trained)
        predictions = loaded_model.predict(self.features)
        self.assertEqual(len(predictions), len(self.labels))

    def test_model_from_pickled_file_classifier(self):
        """Test model loading from a pickled file and its functionality."""
        # Assume 'xgboost_model.pkl' is the pickled model file created earlier
        pickled_model_path = 'classifier_model.pkl'
        loaded_model = ModelFactory.create_model_from_pickled(pickled_model_path)
        self.assertIsInstance(loaded_model, XGBoostModel)
        self.assertIsInstance(loaded_model.model, XGBClassifier)

        # Verify that the loaded model can make predictions (assumes model was trained)
        predictions = loaded_model.predict(self.features)
        self.assertEqual(len(predictions), len(self.labels))

    def test_data_preparing_strategy(self):
        """Test data preparation using the strategy."""
        dmatrix = ToDmatrixStrategy.execute(self.features, self.labels)
        self.assertTrue(isinstance(dmatrix, xgb.DMatrix))

    def test_base_model_manager_singleton_and_cloning(self):
        """Test singleton behavior and cloning of the base model."""
        pickled_model_path = 'classifier_model.pkl'
        loaded_model = ModelFactory.create_model_from_pickled(pickled_model_path)
        BaseModelManager.set_base_model(loaded_model)
        base_model = BaseModelManager.get_instance()
        self.assertIsInstance(base_model, XGBoostModel)

        # Test cloning
        cloned_model = BaseModelManager.clone_base_model()
        self.assertIsInstance(cloned_model, XGBoostModel)
        self.assertIsNot(cloned_model.model, base_model.model)
        self.assertEqual(cloned_model.model_class, base_model.model_class)

if __name__ == '__main__':
    unittest.main()
