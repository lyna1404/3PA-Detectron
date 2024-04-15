import numpy as np
from sklearn.model_selection import GridSearchCV

from ..ModelManager.Models import RandomForestRegressorModel, DecisionTreeRegressorModel
from ..ModelManager.DataPreparingStrategy import ToDataframesStrategy
from .tree_structure import TreeRepresentation

class IPCModel: 
    def __init__(self) -> None:
        params= {'random_state' : 54288}
        self.model = RandomForestRegressorModel(params)

    def optimize(self, param_grid, cv, x, error_prob, sample_weight):
        grid_search = GridSearchCV(estimator=self.model.model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(x, error_prob, sample_weight=sample_weight)
        self.model.set_model(grid_search.best_estimator_)
    
    def train(self, x, Y):
        self.model.train(x, Y)

    def predict(self, x):
        return self.model.predict(x)

class APCModel:
    def __init__(self, features : list, max_depth=None, min_sample_ratio=1) -> None:
        if min_sample_ratio <= 0:
            min_sample_ratio = 1
        else:
            min_sample_ratio = min_sample_ratio / 100
        params = {'max_depth' : max_depth, 'min_samples_leaf' : min_sample_ratio, 'random_state' : 54288}
        self.model = DecisionTreeRegressorModel(params)
        self.treeRepresentation = TreeRepresentation(features=features)
        self.dataPreparationStrategy = ToDataframesStrategy()
        self.features = features
    
    def train(self, X, y):
        self.model.train(X, y)
        df_X, df_y = self.dataPreparationStrategy.execute(column_labels=self.features, features=X, labels=y)
        self.treeRepresentation.head = self.treeRepresentation.build_tree(self.model, df_X, y, 0)

    def predict(self, X, depth=None, min_samples_ratio=0):
        
        df_X, _ = self.dataPreparationStrategy.execute(column_labels=self.features, features=X, labels=None)
        predictions = []
        
        # Loop through each row in X
        for index, row in df_X.iterrows():
            # Make prediction for the current row using the tree
            if self.treeRepresentation.head is not None:
                prediction = self.treeRepresentation.head.assign_node(row, depth, min_samples_ratio)
                predictions.append(prediction)
            else:
                raise ValueError("The Tree Representation has not been initialized, try fitting the APCModel first.")

        
        return np.array(predictions)
    
class MPCModel:
    def __init__(self, IPC_values, APC_values) -> None:
        self.IPC_values = IPC_values
        self.APC_values = APC_values
    
    def predict(self,min_samples_ratio=0):
        if min_samples_ratio >= 0:
            MPC_values = np.minimum(self.IPC_values, self.APC_values) 
        else:
            MPC_values = self.IPC_values
        
        return MPC_values