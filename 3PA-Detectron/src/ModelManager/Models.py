import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from .DataPreparingStrategy import ToDmatrixStrategy, ToNumpyStrategy

class Model:
    """
    Abstract base class for models. Defines the structure that all models should follow.
    """
    def train(self, X, y):
        """
        Train the model on the given dataset.

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

    Attributes:
    params (dict): Parameters for the model.
    model (Model): The trained model instance.
    model_class (str): The class of the model.
    pickled_model (bool): Boolean indicating if a pickled model was provided.
    data_preparation_strategy (ToDmatrixStrategy): The data preparation strategy for the model.
    model_class (str): The class of the model.
    """
    def __init__(self, params_or_model=None, model_class=None):
        """
        Initialize the XGBoostModel either with parameters for a new model or a loaded pickled model.

        :param params_or_model: Either a dictionary of parameters for the booster model 
            or a loaded pickled model.
        :param model_class: Specifies the class of the model if a pickled model is provided. 
            Defaults to xgb.Booster if None.
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
        Ensure that the input data is converted to a DMatrix format, 
            using the defined data preparation strategy.

        :param features: Features array.
        :param labels: Labels array, optional.

        :return: A DMatrix object.
        """
        if not isinstance(features, xgb.DMatrix):
            return self.data_preparation_strategy.execute(features, labels)
        else: 
            return features

    def train(self, X, y):
        """
        Train the model on the provided dataset.

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
        Make predictions with the model for the given input.

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


class RandomForestRegressorModel(Model):
    """
    A concrete implementation of the Model class for RandomForestRegressor models, 
    to be used with Med3pa sub-module.
    
    Attributes:
    params (dict): Parameters for the model.
    model (Model): The trained model instance.
    model_class (str): The class of the model.
    pickled_model (bool): Boolean indicating if a pickled model was provided.
    data_preparation_strategy (ToNumpyStrategy): The data preparation strategy for the model.
    model_class (str): The class of the model.
    

    """
    def __init__(self, params):
        """
        Initialize the RandomForestRegressorModel with a sklearn RandomForestRegressor.

        :param params_or_model:DataPreparingStrategy Either a dictionary of parameters for the booster model 
            or a loaded pickled model.
        :param model_class: Specifies the class of the model if a pickled model is provided. 
            Defaults to xgb.Booster if None.
        """
        self.params = params
        self.model = RandomForestRegressor(**params)
        self.model_class = RandomForestRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def set_model(self, model : RandomForestRegressor):
        """
        Setter for RandomForestRegressor model

        :param model: A regressor model instance to be set 
        :type model: RandomForestRegressor
        """
        self.model = model

    def _ensure_numpy_arrays(self, features, labels=None) -> tuple[np.ndarray,np.ndarray]:
        """
        Ensure that the input data is converted to NumPy array format. 

        :param features: Features data, which can be in various formats like lists,
          Pandas DataFrames, or already in NumPy arrays.
        :param labels: Labels data, optional, similar to features in that it can be in various formats.
        
        :return: The features and labels (if provided) as NumPy arrays.
        :rtype: (np.ndarray, np.ndarray)
        """
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            return self.data_preparation_strategy.execute(features, labels)
        else: 
            return features, labels

    def train(self, X, y):
        """
        Trains the model on the provided dataset.

        :param X: Features for training.
        :param y: Labels for training.

        :raises
            ValueError: If the RandomForestRegressor has not been initialized before training.
        """
        if self.model is None:
                raise ValueError("The RandomForestRegressor has not been initialized.")
        else:
            np_X, np_y = self._ensure_numpy_arrays(X, y)
            self.model.fit(np_X, np_y)      
       
    def predict(self, X):
        """
        Makes predictions with the model for the given input.

        :param X: Features for prediction.

        :return: Predictions made by the model.

        :raises
            ValueError: If the RandomForestRegressor has not been initialized before training.
        """
        
        if self.model is None:
                raise ValueError("The RandomForestRegressor has not been initialized.")
        else:
            np_X, _ = self._ensure_numpy_arrays(X)
            return self.model.predict(np_X)      
        

class DecisionTreeRegressorModel(Model):
    """
    A concrete implementation of the Model class for DecisionTree models, to be used with Med3pa sub-module.

    Attributes:
    params (dict): Parameters for the model.
    model (Model): The trained model instance.
    model_class (str): The class of the model.
    pickled_model (bool): Boolean indicating if a pickled model was provided.
    data_preparation_strategy (ToNumpyStrategy): The data preparation strategy for the model.
    model_class (str): The class of the model.
    """
    def __init__(self, params):
        """
        Initialize the DecisionTreeRegressorModel with a sklearn RandomForestRegressor.

        :param params_or_model: Either a dictionary of parameters for the booster model 
        or a loaded pickled model.
        :param model_class: Specifies the class of the model if a pickled model is provided. 
        Defaults to xgb.Booster if None.
        """
        self.params = params
        self.model = DecisionTreeRegressor(**params)
        self.model_class = DecisionTreeRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def set_model(self, model : DecisionTreeRegressor):
        """
        Setter for DecisionTreeRegressor model

        :param model: A regressor model instance to be set 
        :type model: DecisionTreeRegressor
        """
        self.model = model

    def _ensure_numpy_arrays(self, features, labels=None) -> tuple[np.ndarray,np.ndarray]:
        """
        Ensures that the input data is converted to NumPy array format. 

        :param features: Features data, which can be in various formats like lists, Pandas DataFrames, or already in NumPy arrays.
        :param labels: Labels data, optional, similar to features in that it can be in various formats.
        
        :return: The features and labels (if provided) as NumPy arrays.
        """
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            return self.data_preparation_strategy.execute(features, labels)
        else: 
            return features, labels

    def train(self, X, y):
        """
        Train the model on the provided dataset.

        :param X: Features for training.
        :param y: Labels for training.

        :raises
            ValueError: If the DecisionTreeRegressor has not been initialized before training.
        """
        if self.model is None:
                raise ValueError("The DecisionTreeRegressor has not been initialized.")
        else:
            np_X, np_y = self._ensure_numpy_arrays(X, y)
            self.model.fit(np_X, np_y)      
       
    def predict(self, X):
        """
        Make predictions with the model for the given input.

        :param X: Features for prediction.

        :return: Predictions made by the model.
        
        :raises
            ValueError: If the DecisionTreeRegressor has not been initialized before training.
        """
        
        if self.model is None:
                raise ValueError("The DecisionTreeRegressor has not been initialized.")
        else:
            np_X, _ = self._ensure_numpy_arrays(X)
            return self.model.predict(np_X)      