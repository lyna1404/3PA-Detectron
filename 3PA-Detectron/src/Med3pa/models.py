import numpy as np
from sklearn.model_selection import GridSearchCV

from ..ModelManager.Models import RandomForestRegressorModel, DecisionTreeRegressorModel
from ..ModelManager.DataPreparingStrategy import ToDataframesStrategy
from .tree_structure import TreeRepresentation


class IPCModel: 
    """
    Individualized Predictive Confidence (IPC) model for the regression task.

    This model provides predictions along with individualized confidence estimates for each prediction,
    based on the predictions and error probabilities of a base model.

    Attributes:
        model (RandomForestRegressorModel): The underlying random forest regressor model.

    """ 
    def __init__(self) -> None:
        """
        Initialize the IPCModel with a random state for reproducibility.

        """
        params= {'random_state' : 54288}
        self.model = RandomForestRegressorModel(params)

    def optimize(self, param_grid, cv:int, x, error_prob, sample_weight:np.ndarray):
        """
        Optimize the IPCModel by searching for the best hyperparameters using GridSearchCV.

        :param param_grid: The parameter grid for the GridSearchCV.
        :param cv: The number of cross-validation folds.
        :type cv: int
        :param x: The input features.
        :param error_prob: The error probabilities.
        :param sample_weight: The sample weights.
        :type  sample_weight: np.ndarray

        """
        grid_search = GridSearchCV(estimator=self.model.model, 
                                   param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(x, error_prob, sample_weight=sample_weight)
        self.model.set_model(grid_search.best_estimator_)
    
    def train(self, x:np.ndarray, Y:np.ndarray):
        """
        Train the IPCModel on the given input features (base model predictions) 
            and target values (error probabilities).

        :param x: The input features.
        :type x: np.ndarray
        :param Y: The target values.
        :type Y: np.ndarray

        """
        self.model.train(x, Y)

    def predict(self, x:np.ndarray):
        """
        Make predictions using the IPCModel.

        :param x: The input features for prediction.
        :type x: np.ndarray

        :return: The predicted values.
        :rtype: np.ndarray

        """
        return self.model.predict(x)


class APCModel:
    """
    Agregated Predictive Confidence (APC) model for regression tasks.
    
    This model aggregates predictions and error estimates from the IPC model, 
    grouping similar points detected by the IPC model. 
    It identifies and forms problematic profiles for which the base model doesn't perform well.

    Attributes:
        features (list): The list of feature names.
        model (DecisionTreeRegressorModel): The underlying decision tree regressor model.
        treeRepresentation (TreeRepresentation): The representation of the decision tree.
        dataPreparationStrategy (ToDataframesStrategy): The data preparation strategy.
    """
    def __init__(self, features : list, max_depth:int=None, min_sample_ratio:int=1) -> None:
        """
        Initialize the APCModel with the given features, max depth, and minimum sample ratio.

        :param features: The list of feature names.
        :type features: list
        :param max_depth: The maximum depth of the decision tree (default=None).
        :type max_depth: int
        :param min_sample_ratio: The minimum sample ratio (default=1).
        :type min_sample_ratio: int
        """
        if min_sample_ratio <= 0:
            min_sample_ratio = 1
        else:
            min_sample_ratio = min_sample_ratio / 100
        params = {'max_depth' : max_depth, 
                  'min_samples_leaf' : min_sample_ratio, 
                  'random_state' : 54288}
        self.model = DecisionTreeRegressorModel(params)
        self.treeRepresentation = TreeRepresentation(features=features)
        self.dataPreparationStrategy = ToDataframesStrategy()
        self.features = features
    
    def train(self, X:np.ndarray, y:np.ndarray):
        """
        Train the APCModel on the given input features (X) and target values (IPC predictions).

        :param X: The input features.
        :type X: np.ndarray
        :param y: The target values which are the IPC prediction values.
        :type y: np.ndarray
        """
        self.model.train(X, y)
        df_X, _ = self.dataPreparationStrategy.execute(column_labels=self.features, 
                                                          features=X, labels=y)
        self.treeRepresentation.head = self.treeRepresentation.build_tree(self.model, 
                                                                          df_X, y, 0)

    def predict(self, X:np.ndarray, depth:int=None, min_samples_ratio:int=0):
        """
        Make predictions using the APCModel.

        :param X: The input features for prediction.
        :type X: np.ndarray
        :param depth: The depth of the decision tree (default=None).
        :type depth: int
        :param min_samples_ratio: The minimum sample ratio (default=0).
        :type min_smaples_ratio: int

        :return: The predicted values.
        :rtype: np.ndarray
        """
        df_X, _ = self.dataPreparationStrategy.execute(column_labels=self.features, 
                                                       features=X, labels=None)
        predictions = []
        
        # Loop through each row in X
        for _, row in df_X.iterrows():
            # Make prediction for the current row using the tree
            if self.treeRepresentation.head is not None:
                prediction = self.treeRepresentation.head.assign_node(row, depth, 
                                                                      min_samples_ratio)
                predictions.append(prediction)
            else:
                raise ValueError('''The Tree Representation has not been initialized, 
                                 try fitting the APCModel first.''')

        
        return np.array(predictions)
    

class MPCModel:
    """
    Mixed Predictive Confidence (MPC) model considering both IPC and APC values.

    Attributes:
        IPC_values (np.ndarray): The IPC (Individualized Predictive Confidence) values.
        APC_values (np.ndarray): The APC (Agregated Predictive Confidence) values.
    """
    def __init__(self, IPC_values:np.ndarray, APC_values:np.ndarray) -> None:
        """
        Set the MPCModel with the provided IPC and APC values.

        :param IPC_values: The IPC values.
        :type  IPC_values: np.ndarray
        :param APC_values: The APC values.
        :type  APC_values: np.ndarray
        """
        self.IPC_values = IPC_values
        self.APC_values = APC_values
    
    def predict(self,min_samples_ratio:int=0):
        """
        Predict the MPC values based on the minimum sample ratio.

        the MPCModel takes the minimum of the IPC and APC values to provide
          a conservative estimate of predictive confidence.

        If the minimum sample ratio is non-negative, MPC values are calculated as the element-wise minimum
        of IPC and APC values so that the final values are not overly optimistic.
        Otherwise, only IPC values are returned.

        :param min_samples_ratio: The minimum sample ratio (default=0).
        :type min_samples_ratio: int

        :return: The MPC values.
        :rtype: np.ndarray
        """
        if min_samples_ratio >= 0:
            MPC_values = np.minimum(self.IPC_values, self.APC_values) 
        else:
            MPC_values = self.IPC_values
        
        return MPC_values