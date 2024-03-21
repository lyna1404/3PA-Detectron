import pickle
from Models import Model, XGBoostModel
import xgboost as xgb

class ModelFactory:
    """
    A factory class for creating models with different types, using the factory design pattern.
    It supports creating models based on hyperparameters or loading them from pickled files.
    """
    
    # Maps model type names to the actually supported implementation.
    model_mapping = {
        'XGBoostModel': [xgb.Booster, xgb.XGBClassifier],
    }

    # Maps model type names to their corresponding factory classes.
    factories = {
        'XGBoostModel': lambda: XGBoostFactory(),
    }

    @staticmethod
    def get_factory(model_type: str):
        """
        Retrieves the factory object for the given model type.

        Parameters:
            model_type (str): The type of model for which the factory is to be retrieved.

        Returns:
            An instance of the factory associated with the given model type.

        Raises:
            ValueError: If no factory is available for the given model type.
        """
        factory_initializer = ModelFactory.factories.get(model_type)
        if factory_initializer:
            return factory_initializer()
        else:
            raise ValueError(f"No factory available for model type: {model_type}")
        
    @staticmethod
    def create_model_with_hyperparams(model_type: str, hyperparams: dict) -> Model:
        """
        Creates a model of the specified type with the given hyperparameters.

        Parameters:
            model_type (str): The type of model to create.
            hyperparams (dict): A dictionary of hyperparameters for the model.

        Returns:
            A model instance of the specified type, initialized with the given hyperparameters.
        """
        factory = ModelFactory.get_factory(model_type)
        return factory.create_model_with_hyperparams(hyperparams)

    @staticmethod
    def create_model_from_pickled(pickled_file_path: str) -> Model:
        """
        Creates a model by loading it from a pickled file.

        Parameters:
            pickled_file_path (str): The file path to the pickled model file.

        Returns:
            A model instance loaded from the pickled file.

        Raises:
            IOError: If there is an error loading the model from the file.
            TypeError: If the loaded model is not of a supported type.
        """
        try:
            with open(pickled_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            raise IOError(f"Failed to load the model from {pickled_file_path}: {e}")
        
        for model_type, model_classes in ModelFactory.model_mapping.items():
            if any(isinstance(loaded_model, model_class) for model_class in model_classes):
                factory = ModelFactory.get_factory(model_type)
                return factory.create_model_from_pickled(loaded_model)

        raise TypeError("The loaded model is not of a supported type")

class XGBoostFactory(ModelFactory):
    """
    A factory for creating XGBoost model objects, either from hyperparameters or by loading from pickled files.
    Inherits from ModelFactory and specifies creation methods for XGBoost models.
    """
    
    def create_model_with_hyperparams(self, hyperparams: dict) -> XGBoostModel:
        """
        Creates an XGBoostModel with the given hyperparameters.

        Parameters:
            hyperparams (dict): A dictionary of hyperparameters for the XGBoost model.

        Returns:
            An instance of XGBoostModel initialized with the given hyperparameters.
        """
        return XGBoostModel(params_or_model=hyperparams)

    def create_model_from_pickled(self, loaded_model) -> XGBoostModel:
        """
        Recreates an XGBoostModel from a loaded pickled model.

        Parameters:
            loaded_model: The loaded model object, expected to be an instance of xgb.Booster or xgb.XGBClassifier.

        Returns:
            An instance of XGBoostModel created from the loaded model.

        Raises:
            TypeError: If the loaded model is not a supported implementation of the XGBoost model.
        """
        if isinstance(loaded_model, (xgb.Booster, xgb.XGBClassifier)):
            return XGBoostModel(params_or_model=loaded_model, model_class=type(loaded_model))
        else:
            raise TypeError("Loaded model is not an XGBoost model")
