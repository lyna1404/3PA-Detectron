import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from DataPreparingStrategy import ToDmatrixStrategy

class Model:
    """
    Abstract base class for models. Defines the structure that all models should follow.
    """
    def train(self, X, y):
        """
        Trains the model on the given dataset.

        :param X: Features for training.
        :param y: Labels for training.
        :raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, X):
        """
        Makes predictions using the model for the given input.

        :param X: Features for prediction.
        :return: Predictions made by the model.
        :raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class XGBoostModel(Model):
    """
    A concrete implementation of the Model class for XGBoost models.
    """
    def __init__(self, params_or_model=None, model_class=None):
        """
        Initializes the XGBoostModel either with parameters for a new model or a loaded pickled model.

        :param params_or_model: Either a dictionary of parameters for the booster model or a loaded pickled model.
        :param model_class: Specifies the class of the model if a pickled model is provided. Defaults to xgb.Booster if None.
        """
        self.params = None
        self.model = None
        self.model_class = None
        self.pickled_model = False
        self.data_preparation_strategy = ToDmatrixStrategy()

        if model_class is None:  # The model is initialized using parameters
            self.params = params_or_model
            self.model_class = xgb.Booster
        else:  # A pickled model is provided
            self.model = params_or_model
            self.model_class = model_class
            self.pickled_model = True

    def _ensure_dmatrix(self, features, labels=None):
        """
        Ensures that the input data is converted to a DMatrix format, using the defined data preparation strategy.

        :param features: Features array.
        :param labels: Labels array, optional.
        :return: A DMatrix object.
        """
        return self.data_preparation_strategy.execute(features, labels) if not isinstance(features, xgb.DMatrix) else features

    def train(self, X, y):
        """
        Trains the model on the provided dataset.

        :param X: Features for training.
        :param y: Labels for training.
        :raises
            ValueError: If parameters for xgb.Booster are not initialized before training.
            NotImplementedError: If the model_class is not supported for training.
        """
        
        if self.model_class is xgb.Booster: 
            if not self.params:
                raise ValueError("Parameters for xgb.Booster must be initialized before training.")
            dtrain = self._ensure_dmatrix(X, y)
            self.model = xgb.train(self.params, dtrain)
        elif self.model_class is xgb.XGBClassifier: 
            self.model.fit(X, y)
        else:
            raise NotImplementedError(f"Training not implemented for model class {self.model_class}")

    def predict(self, X):
        """
        Makes predictions with the model for the given input.

        :param X: Features for prediction.
        :return: Predictions made by the model.
        :raises
            ValueError: If the model has not been trained.
            NotImplementedError: If the model_class is not supported for prediction.
        """
        if self.model_class is xgb.Booster:
            if self.model is None:
                raise ValueError("The xgb.Booster model has not been trained.")
            dtest = self._ensure_dmatrix(X)
            return self.model.predict(dtest)
        elif self.model_class is xgb.XGBClassifier:
            if self.model is None:
                raise ValueError("The XGBClassifier model has not been instantiated or trained.")
            return self.model.predict(X)
        else:
            raise NotImplementedError(f"Prediction not implemented for model class {self.model_class}")
