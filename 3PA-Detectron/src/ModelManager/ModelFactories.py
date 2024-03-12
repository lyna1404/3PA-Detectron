import pickle
from Models import Model, XGBoostModel
import xgboost as xgb
from ModelFactories import XGBoostFactory

class ModelFactory:
    model_mapping = {
        'XGBoostModel': xgb.Booster,
    }

    factories = {
        'XGBoostModel': XGBoostFactory(),
    }
    # get the appropriate factory for this model type
    def get_factory(self, model_type):
        if model_type in self.factories:
            return self.factories[model_type]
        else:
            raise ValueError(f"No factory available for model type: {model_type}")
    
    # create a model with a pre-defined configuration
    def create_model_with_hyperparams(self, model_type, hyperparams=None):

        # get the appropriate factory for the model
        factory = self.get_factory(model_type)
        # create the model using the appropriate factory
        model_instance = factory.create_model_with_hyperparams(model_type, hyperparams)
        # return the model
        return model_instance


    def create_model_from_pickled(self, pickled_file_path):
        # Attempt to load the pickled model safely
        try:
            with open(pickled_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            raise IOError(f"Failed to load the model from {pickled_file_path}: {e}")

        # Check if loaded model is an instance of the supported model types
        for model_type, model_class in ModelFactory.model_mapping.items():
            if isinstance(loaded_model, model_class):
                factory = self.get_factory(model_type)
                factory.create_model_from_pickled(loaded_model)
                return model_type, loaded_model

        raise TypeError("The loaded model is not of a supported type")

class XGBoostFactory(ModelFactory):
    def create_model_with_hyperparams(self, hyperparams=None):
        # Logic to create an XGBoost model with hyperparameters
        return XGBoostModel(hyperparams)

    def create_model_from_pickled(self, loaded_model):
        return XGBoostModel(loaded_model)