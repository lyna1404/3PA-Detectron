import xgboost as xgb
from xgboost.sklearn import XGBClassifier

class Model:
    def train(self, X, y):
        pass

    def predict(self, X):
        pass

class XGBoostModel(Model):
    def __init__(self, params_or_model=None, model_class=None):
        """
        Initialize the XGBoostModel using params or a loaded pickled model.
        :param params_or_model: Either a dictionary of parameters for the booster model or a loaded pickled model.
        :param model_class: Class of the model if a pickled model is provided.
        """
        self.params = None
        self.model = None
        self.model_class = None
        self.pickled_model = False
        
        if model_class is None : 
            self.params = params_or_model
            self.model_class = xgb.Booster
            self.pickled_model = False
        else:  
            self.model = params_or_model
            self.model_class = model_class
            self.pickled_model = True

    def train(self, X, y, num_boost_round=10):
        """
        Train the model.
        :param X: Feature matrix as a numpy array or a DMatrix.
        :param y: Labels as a numpy array.
        :param num_boost_round: Number of boosting rounds.
        """
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        """
        Make predictions with the trained model.
        :param X: Feature matrix as a numpy array or a DMatrix.
        :return: A numpy array of predictions.
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

class BaseModelManager(Model):
    __baseModel = None

    @classmethod
    def set_base_model(cls, model):
        if cls.__baseModel is None:
            cls.__baseModel = model
        else:
            raise TypeError("The Base Model has already been initialized")

    @classmethod
    def get_instance(cls):
        if cls.__baseModel is None:
            raise TypeError("The Base Model has not been initialized yet")
        else:
            return cls.__baseModel


