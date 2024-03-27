from ModelManager.Models import DecisionTreeRegressorModel
from pandas import DataFrame, Series
import numpy as np

class TreeRepresentation:
    def __init__(self, features : list) -> None:
        self.features = features
        self.head = None
        self.nb_nodes = 0

    def build_tree(self, dtr : DecisionTreeRegressorModel, X, y, node_id=0):
        
        self.nb_nodes += 1
        left_child = dtr.model.tree_.children_left[node_id]
        right_child = dtr.model.tree_.children_right[node_id]

        node_value = y.mean()
        node_max = y.max()

        node_samples_ratio = dtr.model.tree_.n_node_samples[node_id] / self.dtr.tree_.n_node_samples[0] * 100

        # If we are at a leaf
        if left_child == -1:
            curr_node = _TreeNode(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio,
                              node_id=self.nb_nodes)
            return curr_node

        node_thresh = dtr.model.tree_.threshold[node_id]
        node_feature_id = dtr.model.tree_.feature[node_id]
        node_feature = self.features[node_feature_id]

        curr_node = _TreeNode(value=node_value,
                          value_max=node_max,
                          samples_ratio=node_samples_ratio,
                          threshold=node_thresh,
                          feature=node_feature,
                          feature_id=node_feature_id,
                          node_id=self.nb_nodes)

        curr_node.c_left = self.add_children(left_child,
                                             X=X.loc[X[node_feature] <= node_thresh],
                                             y=y[X[node_feature] <= node_thresh])
        curr_node.c_right = self.add_children(right_child,
                                              X=X.loc[X[node_feature] > node_thresh],
                                              y=y[X[node_feature] > node_thresh])

        return curr_node
    def get_all_profiles(self, min_ca=0, min_samples_ratio=0):
        profiles, nodes_numbers = self.head.get_profile(min_samples_ratio=min_samples_ratio, min_ca=min_ca,
                                                        previous_thresh="*")
        return profiles, nodes_numbers

class _TreeNode:
    def __init__(self, value, value_max, samples_ratio, threshold=None, feature=None, feature_id=None, node_id=0):
        self.c_left = None
        self.c_right = None
        self.value = value
        self.value_max = value_max
        self.samples_ratio = samples_ratio
        self.threshold = threshold
        self.feature = feature
        self.feature_id = feature_id
        self.node_id = node_id

    def assign_node(self, X, depth=None, min_samples_ratio=0):
        if depth == 0 or self.c_left is None:
            return self.value

        if type(X) == DataFrame:
            X_value = X[self.feature]
        elif type(X) == Series:
            X_value = X[self.feature_id]
        else:
            raise TypeError(f"Parameter X is of type {type(X)}, but it must be of type "
                            f"'pandas.DataFrame' or 'pandas.Series'.")

        if depth is not None:
            depth -= 1

        if X_value <= self.threshold:  # If node split condition is true, then left children
            c_node = self.c_left
        else:
            c_node = self.c_right

        if c_node.samples_ratio < min_samples_ratio:  # If not enough samples in child node
            return self.value

        return c_node.assign_node(X, depth, min_samples_ratio)
    
    def get_profile(self, min_samples_ratio, min_ca, previous_thresh=""):
        curr_profile_child = []
        curr_child_nodeid = []
        prev_thresh_separator = " / " if previous_thresh != "" else ""
        temp = 0
        if self.c_left is not None:
            if self.c_left.samples_ratio >= min_samples_ratio:  # self.c_left.value >= min_ca and
                temp += 1
                left_prev_thresh = previous_thresh + f"{prev_thresh_separator}{self.feature}<=" \
                                                     f"{round(self.threshold, 2)}"
                c_prev_str, c_prev_id = self.c_left.get_profile(min_samples_ratio=min_samples_ratio,
                                                                min_ca=min_ca, previous_thresh=left_prev_thresh)
                curr_profile_child += c_prev_str
                curr_child_nodeid += c_prev_id

        if self.c_right is not None:
            if self.c_right.samples_ratio >= min_samples_ratio:  # self.c_right.value >= min_ca and
                temp += 1
                right_prev_thresh = previous_thresh + f"{prev_thresh_separator}{self.feature}>" \
                                                      f"{round(self.threshold, 2)}"
                c_prev_str, c_prev_id = self.c_right.get_profile(min_samples_ratio=min_samples_ratio,
                                                               min_ca=min_ca, previous_thresh=right_prev_thresh)
                curr_profile_child += c_prev_str
                curr_child_nodeid += c_prev_id

        if min_samples_ratio < 0:  # Case where we don't use profiles
            if self.value_max < min_ca and len(curr_profile_child) == 0:
                return [], []
        else:
            if self.value < min_ca and len(curr_profile_child) == 0:
                return [], []

        return [*curr_profile_child, previous_thresh], [*curr_child_nodeid, self.node_id]

    