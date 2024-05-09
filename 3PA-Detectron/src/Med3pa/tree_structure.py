from typing import List,Union
import numpy as np
from pandas import DataFrame, Series

from ..ModelManager.Models import DecisionTreeRegressorModel


class TreeRepresentation:
    """
    Represents the tree structure of MED3PA.

    This class provides methods for building a tree representation.

    Attributes:
        features (list): The list of features used in the tree.
        head (_TreeNode): The root node of the tree.
        nb_nodes (int): The total number of nodes in the tree.
    """
    def __init__(self, features : list) -> None:
        """
        Initialize a TreeRepresentation object.

        :param features: A list of feature names.
        :type features: list
        """
        self.features = features
        self.head = None
        self.nb_nodes = 0

    def build_tree(self, dtr : DecisionTreeRegressorModel, X:Union[DataFrame,Series], y:np.ndarray, 
                    node_id:int=0, path:List[str]=['*']):
        """
        Build a tree representation based on a Regression Tree model.

        :param dtr: A DecisionTreeRegressorModel object.
        :type dtr: DecisionTreeRegressorModel
        :param X: The input data.
        :type X: Union[DataFrame,Series]
        :param y: The target labels.
        :type y:  np.ndarray
        :param node_id: The node ID (default 0).
        :type node_id: int
        :param path: the path from the root to the current node (default [*]),
        :type path: List[str]
        """
        self.nb_nodes += 1
        left_child = dtr.model.tree_.children_left[node_id]
        right_child = dtr.model.tree_.children_right[node_id]

        node_value = y.mean()
        node_max = y.max()

        node_samples_ratio = dtr.model.tree_.n_node_samples[node_id] / dtr.model.tree_.n_node_samples[0] * 100

        # If a leaf node is reached, create a new one:
        if left_child == -1:
            curr_node = _TreeNode(value=node_value, value_max=node_max, 
                                  samples_ratio=node_samples_ratio,
                                  node_id=self.nb_nodes, path=path)
            return curr_node

        node_thresh = dtr.model.tree_.threshold[node_id]
        node_feature_id = dtr.model.tree_.feature[node_id]
        node_feature = self.features[node_feature_id]
        
        # Copy the path of the current node to avoid losing the head
        curr_path = list(path)  
        # Create a new internal node
        curr_node = _TreeNode(value=node_value, value_max=node_max, 
                              samples_ratio=node_samples_ratio,
                              threshold=node_thresh, feature=node_feature, 
                              feature_id=node_feature_id,
                              node_id=self.nb_nodes, path=curr_path)

        # Save the paths of child nodes and recursively build them
        left_path = curr_path + [f"{node_feature} <= {node_thresh}"]
        right_path = curr_path + [f"{node_feature} > {node_thresh}"]
        curr_node.c_left = self.build_tree(dtr, X=X.loc[X[node_feature] <= node_thresh], 
                                           y=y[X[node_feature] <= node_thresh],
                                           node_id=left_child, path=left_path)
        curr_node.c_right = self.build_tree(dtr, X=X.loc[X[node_feature] > node_thresh], 
                                            y=y[X[node_feature] > node_thresh],
                                            node_id=right_child, path=right_path)

        return curr_node

    def get_all_profiles(self, min_ca:int=0, min_samples_ratio:int=0):
        """
        Get all profiles based on specified conditions.

        :param min_ca: The minimum Conditional Accuracy value (default 0).
        :type min_ca: int
        :param min_samples_ratio: The minimum samples ratio (default 0).
        :type min_samples_ratio: int

        :return: A list of profiles.
        :rtype: list
        """
        profiles = self.head.get_profile(min_samples_ratio=min_samples_ratio, min_ca=min_ca)
        return profiles
    

class _TreeNode:
    def __init__(self, value, value_max, samples_ratio, 
                 threshold=None, feature=None, feature_id:int=None, node_id:int=0, path:List[str] = []):
        """
        Initialize a tree node.

        :param value: The value of the node.
        :type value: float
        :param value_max: The maximum value of the node.
        :type value_max: int
        :param samples_ratio: The samples ratio of the node.
        :type samples_ratio: int
        :param threshold: The threshold value for splitting the node (default None).
        :type threshold: float
        :param feature: The feature used for splitting the node (default None).
        :type feature: 
        :param feature_id: The feature ID used for splitting the node (default None).
        :type feature_id: int or None
        :param node_id: The ID of the node (default 0).
        :type node_id: int
        :param path: The path from the root to the node (default []).
        :type path: List[str]
        """
        self.c_left = None
        self.c_right = None
        self.value = value
        self.value_max = value_max
        self.samples_ratio = samples_ratio
        self.threshold = threshold
        self.feature = feature
        self.feature_id = feature_id
        self.node_id = node_id
        self.path = path

    def assign_node(self, X:Union[DataFrame,Series], depth:int=None, min_samples_ratio:int=0):
        """
        Assign a node in the tree.

        :param X: The input data.
        :type X: pd.DataFrame or pd.Series
        :param depth: The depth of the node in the tree (default None).
        :type depth: int
        :param min_samples_ratio: The minimum samples ratio (default 0).
        :type min_samples_ratio: int

        :return: The value of the assigned node.
        :rtype: ?

        :raises TypeError: If `X` is not of type `pandas.DataFrame` or `pandas.Series`.
        """
        if depth == 0 or self.c_left is None:
            return self.value

        if isinstance(X, DataFrame):
            X_value = X[self.feature]
        elif isinstance(X, Series):
            X_value = X.iloc[self.feature_id]
        else:
            raise TypeError(f"Parameter X is of type {type(X)}, but it must be of type "f"'pandas.DataFrame' or 'pandas.Series'.")

        if depth is not None:
            depth -= 1

        # If the node's split condition is true, move to the left
        if X_value <= self.threshold:  
            c_node = self.c_left
        else:
            c_node = self.c_right

        # If there are not enough samples in the child node
        if c_node.samples_ratio < min_samples_ratio:  
            return self.value

        return c_node.assign_node(X, depth, min_samples_ratio)

    def get_profile(self, min_samples_ratio, min_ca):
        """
        Get profile based on specified conditions.

        :param min_samples_ratio: The minimum samples ratio.
        :type min_samples_ratio: float
        :param min_ca: The minimum Conditional Accuracy (CA) value.
        :type min_ca: float
        
        :return: A list of profile information dictionaries.
        :rtype: list[dict]
        """
        profiles = []
        if self.c_left is not None and self.c_left.samples_ratio >= min_samples_ratio:
            # Recursively retrieve profiles from the left child
            profile_info = self.c_left.get_profile(min_samples_ratio, min_ca)   
            profiles.extend(profile_info)

        if self.c_right is not None and self.c_right.samples_ratio >= min_samples_ratio:
            # Recursively retrieve profiles from the right child
            profile_info = self.c_right.get_profile(min_samples_ratio, min_ca)
            profiles.extend(profile_info)

        # Cases where profiles can't be extracted:
        meets_profile_conditions = (
            (min_samples_ratio < 0 and self.value_max < min_ca) or
            (min_samples_ratio >= 0 and self.value < min_ca)
    )

        if meets_profile_conditions and len(profiles) == 0:
            return []
        
        profile_info = {"id" : self.node_id,
                        "path": self.path,
                        "value" : self.value,
                        "ratio" : self.samples_ratio}
        
        return [*profiles, profile_info]
