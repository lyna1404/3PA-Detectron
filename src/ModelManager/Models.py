import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
import numpy as np
from .DataPreparingStrategy import ToDmatrixStrategy, ToNumpyStrategy
from .eval_metrics import metrics_mappings, EvaluationMetric, Accuracy, AveragePrecision, Recall, RocAuc, F1Score, MatthewsCorrCoef, Precision, Sensitivity, Specificity, BalancedAccuracy, PPV, NPV

class Model:
    """
    Abstract base class for models. Defines the structure that all models should follow.
    """
    def train(self, x_train, y_train, x_validation, y_validation, training_parameters):
        """
        Trains the model on the given dataset.

        :param X: Features for training.
        :param y: Labels for training.
        :raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, X, return_proba=False, threshold=0.5):
        """
        Makes predictions using the model for the given input.

        :param X: Features for prediction.
        :return: Predictions made by the model.
        :raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def evaluate(self, X, y, eval_metric, print_results = False):
        """
        Evaluates the model using specified metrics.

        :param X: Features for evaluation.
        :param y: True labels for evaluation.
        :param metrics: metric to use for evaluation.
        :param return_proba: Boolean, determines whether to use probabilities or class labels for evaluation.
        :param sample_weight: Optional array of weights that are assigned to individual samples.
        :return: A dictionary with metric names and their evaluated scores.
        :raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def train_to_disagree(self, x_train, y_train, x_validation, y_validation, x_test, y_test, training_parameters, balance_train_classes, N):
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

    def get_data_strategy(self):
        return self.data_preparation_strategy
    
    def _ensure_dmatrix(self, features, labels=None, weights=None):
        """
        Ensures that the input data is converted to a DMatrix format, using the defined data preparation strategy.

        :param features: Features array.
        :param labels: Labels array, optional.
        :return: A DMatrix object.
        """
        if not isinstance(features, xgb.DMatrix):
            return self.data_preparation_strategy.execute(features, labels, weights)
        else: 
            return features
        
    def validate_params(self, params):
        # Fetch supported parameters for the given model
        valid_params = {
        'eta', 'min_child_weight', 'max_depth', 'max_leaf_nodes', 'gamma',
        'subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
        'lambda', 'alpha', 'tree_method', 'scale_pos_weight', 'objective',
        'learning_rate', 'n_estimators', 'booster', 'verbosity', 'n_jobs',
        'random_state', 'seed', 'missing', 'num_parallel_tree', 'monotone_constraints',
        'interaction_constraints', 'importance_type', 'gpu_id', 'validate_parameters',
        'predictor', 'eval_metric', 'sample_type', 'normalize_type', 'rate_drop',
        'skip_drop', 'base_score', 'nthread', 'device', 
        }
        invalid_keys = [key for key in params if key not in valid_params]
        return {k: v for k, v in params.items() if k in valid_params}
    
    def balance_train_weights(self, y_train):
        _, counts = np.unique(y_train, return_counts=True)
        assert len(counts) == 2, 'Only binary classification is supported'
        c_neg, c_pos = counts[0], counts[1]
        # make sure the average training weight is 1
        pos_weight, neg_weight = 2 * c_neg / (c_neg + c_pos), 2 * c_pos / (c_neg + c_pos)
        train_weights = np.array([pos_weight if label == 1 else neg_weight for label in y_train])
        return train_weights

    def train(self, x_train, y_train, x_validation, y_validation, training_parameters, balance_train_classes):
        """
        Trains the model on the provided dataset.

        :param X: Features for training.
        :param y: Labels for training.
        :raises
            ValueError: If parameters for xgb.Booster are not initialized before training.
            NotImplementedError: If the model_class is not supported for training.
        """
        # if additional training_params are provided
        if training_parameters is not None:
            # Validate and update model parameters
            valid_training_params = self.validate_params(training_parameters)
            if self.params is not None:
                self.params.update(valid_training_params)
            else:
                self.params = valid_training_params
            # balance the dataset if balance_train_classes is set to True, 
            # if False, attempt to extract the training weights if provided, else set the weigths to 1
            if balance_train_classes:
                weights = self.balance_train_weights(y_train=y_train)
            else:
                weights = training_parameters.get('training_weights', np.ones_like(y_train))
            # attempt to extract the evaluation_metrics if provided, else evaluate the model using Accuracy
            evaluation_metrics = training_parameters.get('custom_eval_metrics', ["Accuracy"])
            # attempt to extract the num_boosting_rounds if provided, else set to 10
            num_boost_rounds = training_parameters.get('num_boost_rounds', 10)
        
        # handle the case where the params are not initialized
        if not self.params:
                raise ValueError("Parameters must be initialized before training.")
            

        # if the model is an xgb.Booster
        if self.model_class is xgb.Booster:
            # the params cannot be uninitialized when training an xgb.Booster
            if not self.params:
                raise ValueError("Parameters for xgb.Booster must be initialized before training.")
            
            # prepare training and validation matrices
            dtrain = self._ensure_dmatrix(x_train, y_train, weights)
            dval = self._ensure_dmatrix(x_validation, y_validation)
            # train the Booster model
            self.model = xgb.train(self.params, dtrain,
                                num_boost_round=num_boost_rounds,
                                evals=[(dval, 'eval')], verbose_eval=False)
            # evaluate the model on the validation data
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics, print_results=True)

        # if the model is an xgb.XGBClassifier
        elif self.model_class is xgb.XGBClassifier:
            self.model = self.model_class(**self.params)
            self.model.fit(x_train, y_train, sample_weight = weights, eval_set=[(x_validation, y_validation)])
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics, print_results=True)
        else:
            raise NotImplementedError(f"Training not implemented for model class {self.model_class}")

    def predict(self, X, return_proba=False, threshold=0.5):
        # handle the case where the model has not been trained or initialized
        if self.model is None:
            raise ValueError(f"The {self.model_class.__name__} model has not been initialized.")

        # if the model is a Booster
        if self.model_class is xgb.Booster:
            # convert test data to a matrix
            dtest = self._ensure_dmatrix(X)
            preds = self.model.predict(dtest)

        # if the model is an
        elif self.model_class is xgb.XGBClassifier:
            if hasattr(self.model, 'predict_proba') and return_proba:
                preds = self.model.predict_proba(X)
            else:
                preds = self.model.predict(X)
        else:
            raise NotImplementedError(f"Prediction not implemented for model class {self.model_class}")

        if return_proba:
            return preds
        else:
            return (preds > threshold).astype(int)  
            
    def evaluate(self, X, y, eval_metrics, print_results = False):

        # Check if model is set
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        # Predict using the model
        probs = self.predict(X, return_proba=True)
        if probs.ndim == 1:
            preds =  (probs > 0.5).astype(int)  # For binary classification
        else:
            raise ValueError("Only binary classification is supported for this version.")
        evaluation_results = {}
        for metric_name in eval_metrics:
            metric = metrics_mappings.get(metric_name, None)
            if metric is not None:
                if metric is RocAuc:
                    evaluation_results[metric_name] = metric.calculate(y, probs)
                elif metric is AveragePrecision:
                    evaluation_results[metric_name] = metric.calculate(y, probs)
                else:
                    evaluation_results[metric_name] = metric.calculate(y, preds)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")
        if print_results:
            self.print_evaluation_results(results=evaluation_results)
        return evaluation_results
    
    def print_evaluation_results(self, results):
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")

    def train_to_disagree(self, x_train, y_train, x_validation, y_validation, x_test, y_test, training_parameters, balance_train_classes, N):
        
        # if additional training_params are provided
        if training_parameters is not None:
            # Validate and update model parameters
            valid_training_params = self.validate_params(training_parameters)
            if self.params is not None:
                self.params.update(valid_training_params)
            else:
                self.params = valid_training_params
            # balance the training dataset if balance_train_classes is set to True, 
            # if False, attempt to extract the training weights if provided, else set the weigths to 1
            if balance_train_classes:
                training_weights = self.balance_train_weights(y_train=y_train)
            else:
                training_weights = training_parameters.get('training_weights', np.ones_like(y_train))

            # attempt to extract the evaluation_metrics if provided, else evaluate the model using Accuracy
            evaluation_metrics = training_parameters.get('custom_eval_metrics', ["Accuracy"])
            # attempt to extract the num_boosting_rounds if provided, else set to 10
            num_boost_rounds = training_parameters.get('num_boost_rounds', 10)
        
        # handle the case where the params are not initialized
        if not self.params:
                raise ValueError("Parameters must be initialized before training.")
            
        # prepare the data, labels and weights for training to disagree
        data=np.concatenate([x_train, x_test])
        label=np.concatenate([y_train, 1 - y_test])
        weight=np.concatenate([training_weights, 1 / (N + 1) * np.ones(N)])

        # if the model is an xgb.Booster
        if self.model_class is xgb.Booster:
            # the params cannot be uninitialized when training an xgb.Booster
            if not self.params:
                raise ValueError("Parameters for xgb.Booster must be initialized before training.")
            
            # prepare training and validation matrices
            dtrain = self._ensure_dmatrix(data, label, weight)
            dval = self._ensure_dmatrix(x_validation, y_validation)
            # train the Booster model
            self.model = xgb.train(self.params, dtrain,
                                num_boost_round=num_boost_rounds,
                                evals=[(dval, 'eval')], verbose_eval=False)
            # evaluate the model on the validation data
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics)

        # if the model is an xgb.XGBClassifier
        elif self.model_class is xgb.XGBClassifier:
            self.model = self.model_class(**self.params)
            self.model.fit(data, label, sample_weight=weight, eval_set=[(x_validation, y_validation)])
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics)
        else:
            raise NotImplementedError(f"Training not implemented for model class {self.model_class}")

    
class RandomForestRegressorModel(Model):
    """
    A concrete implementation of the Model class for RandomForestRegressor models, to be used with Med3pa sub-module.
    """
    def __init__(self, params):
        """
        Initializes the RandomForestRegressorModel with a sklearn RandomForestRegressor.

        :param params_or_model: Either a dictionary of parameters for the booster model or a loaded pickled model.
        :param model_class: Specifies the class of the model if a pickled model is provided. Defaults to xgb.Booster if None.
        """
        self.params = params
        self.model = RandomForestRegressor(**params)
        self.model_class = RandomForestRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def set_model(self, model : RandomForestRegressor):
        self.model = model
    def _ensure_numpy_arrays(self, features, labels=None):
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
    """
    def __init__(self, params):
        """
        Initializes the DecisionTreeRegressorModel with a sklearn RandomForestRegressor.

        :param params_or_model: Either a dictionary of parameters for the booster model or a loaded pickled model.
        :param model_class: Specifies the class of the model if a pickled model is provided. Defaults to xgb.Booster if None.
        """
        self.params = params
        self.model = DecisionTreeRegressor(**params)
        self.model_class = DecisionTreeRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def set_model(self, model : DecisionTreeRegressor):
        self.model = model
    def _ensure_numpy_arrays(self, features, labels=None):
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
        Trains the model on the provided dataset.

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
        Makes predictions with the model for the given input.

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