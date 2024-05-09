import numpy as np
from typing import List, Dict, Any

from ..ModelManager.eval_metrics import  AveragePrecision, RocAuc
from .tree_structure import TreeRepresentation


class MDRCalculator:
    """
    Calculates different general and profile specific evaluation metrics by declaration rate (DR)
    
    DR is the percentage of predictions that are declared positive 
    (e.g., as belonging to a certain class) based on a confidence threshold.
    """
    @staticmethod
    def eval_metrics_by_dr(Y_target:np.ndarray, Y_predicted:np.ndarray, predicted_prob:np.ndarray, 
                            predicted_accuracies:np.ndarray, evalmetrics_list : list):
        """
        Calculate metrics by declaration rate.

        :param Y_target: The target values.
        :type Y_target: np.ndarray
        :param Y_predicted: The predicted values.
        :type Y_predicted: np.ndarray
        :param predicted_prob: The predicted probabilities.
        :type predicted_prob: np.ndarray
        :param predicted_accuracies: The predicted accuracies.
        :type predicted_accuracies: np.ndarray
        :param evalmetrics_list: A list of evaluation metric objects.
        :type evalmetrics_list: list

        :return: A dictionary containing metrics by detection rate.
        :rtype: dict
        """
        metrics_by_dr = {}
        sorted_accuracies = np.sort(predicted_accuracies)
        last_dr_values = {}
        last_min_confidence_level = -1
        for dr in range(100, 0, -1):
            min_confidence_level = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
            if last_min_confidence_level != min_confidence_level :
                last_min_confidence_level = min_confidence_level 
                dr_values = {'min_confidence_level' : min_confidence_level}
                dr_values['PopulationPercentage'] = sum(predicted_accuracies>=min_confidence_level)/len(Y_target)
                # Calculate metrics using the provided metric objects
                metrics_dict = {}
                for metric in evalmetrics_list:
                    if isinstance(metric, RocAuc):
                        metrics_dict['RocAuc'] = metric.calculate(Y_target[predicted_accuracies >= min_confidence_level],
                                                                    predicted_prob[predicted_accuracies >= min_confidence_level])
                    elif isinstance(metric, AveragePrecision):
                        metrics_dict['AveragePrecision'] = metric.calculate(Y_target[predicted_accuracies >= min_confidence_level],
                                                                            predicted_prob[predicted_accuracies >= min_confidence_level])
                    else:
                        metrics_dict[metric.__class__.__name__] = metric.calculate(Y_target[predicted_accuracies >= min_confidence_level],
                                                                                            Y_predicted[predicted_accuracies >= min_confidence_level])
                
                dr_values['Metrics'] = metrics_dict
                last_dr_values = dr_values
                metrics_by_dr[dr] = dr_values
            else : 
                metrics_by_dr[dr] = last_dr_values
        return metrics_by_dr
    
    # Should change names
    @staticmethod
    def _list_difference_by_key(list1:List[Dict[str, Any]], list2:List[Dict[str, Any]], key:str='id'):
        """
        Find the difference between two lists of dictionaries by a specified key.

        :param list1: The first list of dictionaries.
        :type list1: List[Dict[str,Any]]
        :param list2: The second list of dictionaries.
        :type list2: List[Dict[str,Any]]
        :param key: The key to use for comparison (default is 'id').
        :type key: str

        :return: The list of dictionaries unique to list1.
        :rtype: list[Dict[str, Any]]
        """
        set1 = {d[key] for d in list1 if key in d}
        set2 = {d[key] for d in list2 if key in d}

        unique_to_list1 = set1 - set2
        difference_list1 = [d for d in list1 if d.get(key) in unique_to_list1]

        return difference_list1
    
    @staticmethod
    def calc_profiles_by_dr(tree:TreeRepresentation, predicted_accuracies:np.ndarray, min_samples_ratio:int=0):
        """
        Calculate profiles by decaration rate.

        :param tree: The tree representation.
        :type tree: TreeRepresentation
        :param predicted_accuracies: The predicted accuracies.
        :type predicted_accuracies: np.ndarray
        :param min_samples_ratio: The minimum samples ratio (default is 0).
        :type min_samples_ratio: int

        :return: A dictionary containing profiles by declaration rate.
        :rtype: (dict,dict)
        """
        profiles_by_dr = {}
        lost_profiles_by_dr = {}
        sorted_accuracies = np.sort(predicted_accuracies)
        last_dr_values = {}
        last_profiles = tree.get_all_profiles(sorted_accuracies[0], 
                                              min_samples_ratio)
        last_min_confidence_level = -1

        for dr in range(100, 0, -1):
            min_confidence_level = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
            if last_min_confidence_level != min_confidence_level :
                last_min_confidence_level = min_confidence_level 
                dr_values = {'min_confidence_level' : min_confidence_level}
                profiles = tree.get_all_profiles(min_confidence_level, min_samples_ratio)
                dr_values['Profiles'] = profiles
                if last_profiles != profiles:
                    temp = MDRCalculator._list_difference_by_key(last_profiles, profiles, key='id')
                    lost_profiles_by_dr[dr] = {"min_confidence_level" : min_confidence_level, 
                                               "lost_profiles" : temp}
                last_profiles = profiles
                last_dr_values = dr_values
                profiles_by_dr[dr] = dr_values
            else : 
                profiles_by_dr[dr] = last_dr_values
        return profiles_by_dr, lost_profiles_by_dr
    
    @staticmethod
    def _filter_by_profile(X:np.ndarray, Y_true:np.ndarray, predicted_prob:np.ndarray, 
                            Y_pred:np.ndarray, mpc_values:np.ndarray, features: List[str], path:List[str]):
        """
        Filter data based on a given profile path.

        :param X: The input data.
        :type X: np.ndarray
        :param Y_true: The true labels.
        :type Y_true: np.ndarray
        :param predicted_prob: The predicted probabilities.
        :type predicted_prob: np.ndarray
        :param Y_pred: The predicted values.
        :type Y_pred: np.ndarray
        :param mpc_values: The MPCModel values.
        :type mpc_values: np.ndarray
        :param features: The Dataset features.
        :type features: list of strings
        :param path: The profile path.
        :type path: List[str]

        :return: The filtered data.
        :rtype: (np.ndarray,np.ndarray,np.ndarray,np.ndarray)
        """

        # Start with a mask that selects all rows
        mask = np.ones(len(X), dtype=bool)
        
        for condition in path:
            if condition == '*':
                continue  # Skip the root node indicator
            
            # Parse the condition string
            column_name, operator, value_str = condition.split(' ')
            column_index = features.index(column_name)  # Map feature name to index
            value = float(value_str) if value_str.replace('.', '', 1).isdigit() else value_str
            
            # Apply the condition to update the mask
            if operator == '>':
                mask &= X[:, column_index] > value
            elif operator == '<':
                mask &= X[:, column_index] < value
            elif operator == '>=':
                mask &= X[:, column_index] >= value
            elif operator == '<=':
                mask &= X[:, column_index] <= value
            elif operator == '==':
                mask &= X[:, column_index] == value
            elif operator == '!=':
                mask &= X[:, column_index] != value
            else:
                raise ValueError(f"Unsupported operator '{operator}' in condition '{condition}'.")
            
        # Filter the data
        filtered_x = X[mask]
        filtered_y_true = Y_true[mask]
        filtered_prob = predicted_prob[mask]
        filtered_y_pred = Y_pred[mask]
        filtered_mpc_values = mpc_values[mask]
        return filtered_x, filtered_y_true, filtered_prob, filtered_y_pred, filtered_mpc_values
    
    @staticmethod
    def eval_metrics_by_profile(Y_true:np.ndarray, predicted_prob:np.ndarray, 
                            total_population:int, node_population:int, metrics_list:List[str]):
        """
        Calculate evaluation metrics by profile.

        :param Y_true: The true values.
        :type Y_true: np.ndarray
        :param predicted_prob: The predicted probabilities.
        :type predicted_prob: np.ndarray
        :param total_population: The total population.
        :type total_population: int
        :param node_population: The node population.
        :type node_population: int
        :param metrics_list: A list of metric objects.
        :type metrics_list: List[str]

        :return: A dictionary containing metrics by profile.
        :rtype: dict
        """
        metrics_dict = {}
        for metric in metrics_list:
            if isinstance(metric, RocAuc):
                metrics_dict['RocAuc'] = metric.calculate(Y_true, predicted_prob)
            elif isinstance(metric, AveragePrecision):
                metrics_dict['AveragePrecision'] = metric.calculate(Y_true, predicted_prob)
            else:
                metrics_dict[metric.__class__.__name__] = metric.calculate(Y_true, predicted_prob)
        perc_node = len(Y_true) * 100 / node_population
        perc_pop = len(Y_true) * 100 / total_population
        metrics_dict['Node%'] = perc_node
        metrics_dict['Population%'] = perc_pop
        return metrics_dict


class Profile:
    """
    Represents a profile containing evaluation metrics and values associated with a specific node.

    Attributes:
        node_id (int): An identifier for the node.
        path (str): The path (conditions) leading to the node in the tree.
        mean_value (float): The average uncertainty value of the node.
        metrics (dict, optional): Additional metrics.
    """

    def __init__(self, node_id, path, mean_value, metrics=None):
        """
        Initialize a new instance of the Profile class.

        :param node_id: An identifier for the node.
        :param path: The path (conditions) leading to the node in the tree.
        :param mean_value: The average uncertainty value of the node.
        :param metrics: Additional metrics (default is None).
        """
        self.node_id = node_id
        self.path = path
        self.mean_value = mean_value
        self.metrics = metrics

    def to_dict(self):
        """
        Convert the Profile instance into a dictionary format suitable for serialization.

      
        :return: A dictionary representation of the Profile instance including:
            the node ID, path, mean value, and metrics.
        :rtype: dict
        """
        profile = {
            'id': self.node_id,
            'path': self.path,
            'value': self.mean_value,
            'metrics': self.metrics
        }
        return profile

    
class ProfilesManager:
    """
    Manages the records of profiles and lost profiles based on declaration rates and minimal samples ratio
    
    Preforms insertion,retrieval, and transformation of profile data, 
    as well as the extraction of the profiles from a computed tree representation 
    and the calculation of the metrics

    Attributes:
        profiles_records (dict): A nested dictionary storing profiles organized by sample ratio and dr values.
        lost_profiles_records (dict): A nested dictionary storing lost profiles organized similarly 
            to profiles_records.
        features (list): A list of features used in the profiles.
    """
    def __init__(self, features) -> None:
        """
        Initialize the ProfilesManager with the specified features.

        
            :param features: features that will be used with the profiles.
        """
        self.profiles_records = {}
        self.lost_profiles_records ={}
        self.features = features
    
    def insert_profiles(self, dr, min_samples_ratio, profiles : list):
        """
        Add profiles to the records under a specific declaration rate value and a minimum sample ratio.

        :param dr: Declaration rate value.
        :param min_samples_ratio: Minimum samples ratio.
        :param profiles: profiles to be stored.
        :type profiles: list
        """
        if min_samples_ratio not in self.profiles_records:
            self.profiles_records[min_samples_ratio] = {}
        self.profiles_records[min_samples_ratio][dr] = profiles
    
    def insert_lost_profiles(self, dr, min_samples_ratio, profiles : list):
        """
        Add lost profiles to the records under a specific declaration rate value and a minimum sample ratio.

        :param dr: Decision rate value.
        :param min_samples_ratio: Minimum samples ratio.
        :param profiles: lost profiles to be stored.
        :type profiles: list
        """
        if min_samples_ratio not in self.lost_profiles_records:
            self.lost_profiles_records[min_samples_ratio] = {}
        self.lost_profiles_records[min_samples_ratio][dr] = profiles

    def get_profiles(self, min_samples_ratio =None, dr=None):
        """
        Retrieve profiles based on the specified minimum sample ratio and the dr value.

        :param min_samples_ratio: Minimum samples ratio.
        :param dr: Decision rate value.

        :return: Profiles corresponding to the specified filters.
        :rtype: dict
        """
        if min_samples_ratio is not None:
            if dr is not None:
                if min_samples_ratio not in self.profiles_records:
                    raise ValueError("the profiles for this min_samples_ratio has not been calculated yet!")
                else:
                    return self.profiles_records[min_samples_ratio][dr]
            else:
                return self.profiles_records[min_samples_ratio]
        else:
            return self.profiles_records
        
    def get_lost_profiles(self, min_samples_ratio=None, dr=None):
        """
        Retrieve profiles based on the specified minimum sample ratio and the dr value.

        :param min_samples_ratio: Minimum samples ratio.
        :param dr: Decision rate value.
        :return: Profiles corresponding to the specified filters.
        :rtype: list
        """
        if min_samples_ratio is not None:
            if dr is not None:
                if min_samples_ratio not in self.lost_profiles_records:
                    raise ValueError("the lost profiles for this min_samples_ratio has not been calculated yet!")
                else:
                    return self.lost_profiles_records[min_samples_ratio][dr]
            else:
                return self.lost_profiles_records[min_samples_ratio]
        else:
            return self.lost_profiles_records
        
    def transform_to_profiles(profiles_list : list, to_dict:bool=True):
        """
         Transform a list of profile data into instances of the Profile class or dictionaries.

        :param profiles_list: List of dictionaries with keys 'id', 'path', and 'value'.
        :type profiles_list: list[dict]
        :param to_dict: Flag to determine if Profile instances should be returned as dictionaries (default: True).
        :type to_dict: bool
        :return: List of Profile instances or dictionaries based on the `to_dict` parameter.
        :rtype: list[Union[Profile, dict]]
        """
        profiles = []
        for profile in profiles_list:
            if to_dict:
                profile_ins = Profile(profile['id'], profile['path'], profile['value']).to_dict()
            else:
                profile_ins = Profile(profile['id'], profile['path'], profile['value'])

            profiles.append(profile_ins)
        return profiles
            
    def calc_profiles(self, tree: TreeRepresentation, predicted_accuracies, ratio_start, ratio_end, ratio_step):
        """
        Calculate profiles for different declaration rates (dr) and minimum sample ratios by
        assessing changes in profiles across confidence levels derived from predicted accuracies.

        :param tree: A tree structure from which profiles are generated.
        :type tree: TreeRepresentation
        :param predicted_accuracies: An array of predicted accuracy values to sort 
            and use for thresholding profiles.
        :type predicted_accuracies: numpy.ndarray
        :param ratio_start: The starting point for the range of minimum sample ratios.
        :type ratio_start: int
        :param ratio_end: The end point for the range of minimum sample ratios (exclusive).
        :type ratio_end: int
        :param ratio_step: The increment between each sample ratio in the range.
        :type ratio_step: int

        Modifies:
            - self.profiles_records: Updates with new profiles for each combination of 
                declaration rate and sample ratio.
            - self.lost_profiles_records: Updates with lost profiles for the same combinations.
        """
        profiles_by_dr = {}
        lost_profiles_by_dr = {}
        sorted_accuracies = np.sort(predicted_accuracies)
        last_dr_values = {}

        for min_samples_ratio in range(ratio_start, ratio_end, ratio_step):
            last_profiles = tree.get_all_profiles(sorted_accuracies[0], min_samples_ratio)
            last_min_confidence_level = -1
            last_dr=100
            for dr in range(100, -1, -1):
                if dr==0:
                    min_confidence_level = 1.01
                else:
                    min_confidence_level = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
                if last_min_confidence_level != min_confidence_level:
                    last_min_confidence_level = min_confidence_level
                    profiles = tree.get_all_profiles(min_confidence_level, min_samples_ratio)
                    profiles_ins = ProfilesManager.transform_to_profiles(profiles, True)
                    self.insert_profiles(dr, min_samples_ratio, profiles_ins)
                    if last_profiles != profiles:
                        lost_profiles = MDRCalculator._list_difference_by_key(last_profiles, profiles, key='id')
                        lost_profiles_ins = ProfilesManager.transform_to_profiles(lost_profiles, True)
                        lost_profiles_by_dr[dr] = {"min_confidence_level": min_confidence_level, 
                                                   "lost_profiles": lost_profiles}
                        self.insert_lost_profiles(last_dr-1, min_samples_ratio, lost_profiles_ins)
                    else:
                        self.insert_lost_profiles(dr, min_samples_ratio, [])
                    last_dr = dr
                    last_profiles = profiles
                    last_profiles_ins = profiles_ins
                else:
                    profiles_by_dr[dr] = last_dr_values
                    self.insert_profiles(dr, min_samples_ratio, last_profiles_ins)
                    self.insert_lost_profiles(dr, min_samples_ratio, [])

    def _filter_by_profile(self, X, Y_true, predicted_prob, Y_pred, mpc_values, path):
        """
        Filters datasets based on a given path of conditions derived from a profile. 
        
        This method allows for the selection of subsets of data corresponding to 
           a specific criteria defined in the path.

        :param X: The feature dataset from which rows are to be selected.
        :type X: np.ndarray
        :param Y_true: The true labels corresponding to the dataset X.
        :type Y_true: np.ndarray
        :param predicted_prob: The predicted probabilities corresponding to the dataset X.
        :type predicted_prob: np.ndarray
        :param Y_pred: The predicted labels corresponding to the dataset X.
        :type Y_pred: np.ndarray
        :param mpc_values: The mpc values corresponding to the dataset X.
        :type mpc_values: np.ndarray
        :param path: A list of conditions defining the path to filter by,
            with each condition formatted as "column_name operator value".
        :type path: list

        :return: Contains filtered versions of X, Y_true, predicted_prob, 
            Y_pred, and mpc_values based on the path conditions.
        :rtype: tuple

        :raises ValueError:
            If an unsupported operator is included in any condition.
        """

        # Start with a mask that selects all rows
        mask = np.ones(len(X), dtype=bool)
        
        for condition in path:
            if condition == '*':
                continue  # Skip the root node indicator

            # Parse the condition string
            column_name, operator, value_str = condition.split(' ')
            column_index = self.features.index(column_name)  # Map feature name to index
            try:
                value = float(value_str)
            except ValueError:
                # If conversion fails, the string is not a number. Handle it appropriately.
                value = value_str  # If it's supposed to be a string, leave it as string
                        
            # Apply the condition to update the mask
            if operator == '>':
                mask &= X[:, column_index] > value
            elif operator == '<':
                mask &= X[:, column_index] < value
            elif operator == '>=':
                mask &= X[:, column_index] >= value
            elif operator == '<=':
                mask &= X[:, column_index] <= value
            elif operator == '==':
                mask &= X[:, column_index] == value
            elif operator == '!=':
                mask &= X[:, column_index] != value
            else:
                raise ValueError(f"Unsupported operator '{operator}' in condition '{condition}'.")

        # Filter the data
        filtered_x = X[mask]
        filtered_y_true = Y_true[mask]
        filtered_prob = predicted_prob[mask]
        filtered_y_pred = Y_pred[mask]
        filtered_mpc_values = mpc_values[mask]
        return filtered_x, filtered_y_true, filtered_prob, filtered_y_pred, filtered_mpc_values

    def calc_metrics_by_profiles(self, all_x, all_y_true, all_pred_prob, 
                                   all_y_pred, all_mpc_values, metrics_list):
        """
        Calculate various metrics for different profiles and declaration rates based on the provided datasets.
        
        This method filters the dataset for each profile according to specific conditions, 
        evaluates given metrics, and appends these metrics to each profile.

        :param all_x: The complete set of features for all observations.
        :type all_x: np.ndarray
        :param all_y_true: The true labels for all observations.
        :type all_y_true: np.ndarray
        :param all_pred_prob: The predicted probabilities for each observation.
        :type all_pred_prob: np.ndarray
        :param all_y_pred: The predicted labels for each observation.
        :type all_y_pred: np.ndarray
        :param all_mpc_values: The model prediction confidence values for all observations.
        :type all_mpc_values: np.ndarray
        :param metrics_list: A list of metric objects that implement a method for evaluating the dataset.
        :type metrics_list: list

        Modifies:
            Each profile in the profiles_records dictionary is updated to include calculated metrics under a 'metrics' key.
        """
        for min_samp_ratio, dr_dict in self.profiles_records.items():
            for dr, profiles in dr_dict.items():
                sorted_accuracies = np.sort(all_mpc_values)
                if(dr==0):
                    min_confidence_level = 1.01
                else:
                    min_confidence_level = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
                for profile in profiles:
                    x, y_true, pred_prob, y_pred, mpc_values = self._filter_by_profile(all_x, all_y_true, all_pred_prob, 
                    all_y_pred, all_mpc_values, profile['path'])
                    metrics_dict = {}
                    for metric in metrics_list:
                        if isinstance(metric, RocAuc):
                            metrics_dict['RocAuc'] = metric.calculate(y_true[mpc_values >= min_confidence_level], 
                                                                      pred_prob[mpc_values >= min_confidence_level])
                        elif isinstance(metric, AveragePrecision):
                            metrics_dict['AveragePrecision'] = metric.calculate(y_true[mpc_values >= min_confidence_level], 
                                                                                pred_prob[mpc_values >= min_confidence_level])
                        else:
                            metrics_dict[metric.__class__.__name__] = metric.calculate(y_true[mpc_values >= min_confidence_level], 
                                                                                       y_pred[mpc_values >= min_confidence_level])
                    
                    perc_node = len(y_true) * 100 /len(y_true)
                    perc_pop = len(y_true) * 100 / len(all_y_true)
                    metrics_dict['Node%'] = perc_node
                    metrics_dict['Population%'] = perc_pop
                    mean_ca = np.mean(mpc_values[mpc_values >= min_confidence_level]) * 100 if \
                    mpc_values[mpc_values >= min_confidence_level].size > 0 \
                    else np.NaN
                    pos_class_occurence = np.sum(y_true[mpc_values >= min_confidence_level]) / len(y_true[mpc_values >= min_confidence_level]) * 100 if \
                    len(y_true[mpc_values >= min_confidence_level]) > 0 \
                    else np.NaN
                    metrics_dict['Mean CA'] = mean_ca
                    metrics_dict['Positive%'] = pos_class_occurence
                    profile['metrics'] = metrics_dict


