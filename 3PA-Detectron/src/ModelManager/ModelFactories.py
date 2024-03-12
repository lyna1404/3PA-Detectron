import pickle
from Models import Model, XGBoostModel, BaseModelManager
import xgboost as xgb

class ModelFactory:
    model_mapping = {
        'XGBoostModel': [xgb.Booster, xgb.XGBClassifier],
    }

    factories = {
        'XGBoostModel': 'XGBoostFactory',
    }
    
    # get the appropriate factory for this model type
    def get_factory(self, model_type):
        if model_type in self.factories:
            return self.factories[model_type]
        else:
            raise ValueError(f"No factory available for model type: {model_type}")

    def create_model_with_hyperparams(self, model_type, hyperparams : dict):
        factory_name = self.get_factory(model_type)
        factory = globals()[factory_name]()
        return factory.create_model_with_hyperparams(hyperparams)

    def create_model_from_pickled(self, pickled_file_path):
        try:
            with open(pickled_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            raise IOError(f"Failed to load the model from {pickled_file_path}: {e}")

        for model_type, model_classes in ModelFactory.model_mapping.items():
            
            for model_class in model_classes :
                if isinstance(loaded_model, model_class):
                    print("this is the model class", model_class)
                    factory_name = self.get_factory(model_type)
                    print("this is the factory name", factory_name)
                    factory = globals()[factory_name]()
                    return factory.create_model_from_pickled(pickled_file_path, model_class)

        raise TypeError("The loaded model is not of a supported type")

    
class XGBoostFactory(ModelFactory):
    def create_model_with_hyperparams(self, hyperparams):
        return XGBoostModel(hyperparams)

    def create_model_from_pickled(self, loaded_model, model_class):
        return XGBoostModel(loaded_model, model_class)