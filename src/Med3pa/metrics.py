import numpy as np
from ..ModelManager.eval_metrics import EvaluationMetric, Accuracy, AveragePrecision, Recall, RocAuc, F1Score, MatthewsCorrCoef, Precision, Sensitivity, Specificity, BalancedAccuracy, PPV, NPV
from pprint import pprint
from .tree_structure import TreeRepresentation
from .Models import IPCModel, APCModel, MPCModel
from .uncertainty import UncertaintyCalculator, AbsoluteError

class MDRCalculator:
    @staticmethod
    def calc_metrics_by_dr(Y_target, Y_predicted, predicted_prob, predicted_accuracies, metrics_list : list):
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
                for metric in metrics_list:
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
    
    @staticmethod
    def _list_difference_by_key(list1, list2, key='id'):
        set1 = {d[key] for d in list1 if key in d}
        set2 = {d[key] for d in list2 if key in d}

        unique_to_list1 = set1 - set2

        difference_list1 = [d for d in list1 if d.get(key) in unique_to_list1]

        return difference_list1
    
    @staticmethod
    def calc_profiles_by_dr(tree:TreeRepresentation, predicted_accuracies, min_samples_ratio=0):
        profiles_by_dr = {}
        lost_profiles_by_dr = {}
        sorted_accuracies = np.sort(predicted_accuracies)
        last_dr_values = {}
        last_profiles = tree.get_all_profiles(sorted_accuracies[0], min_samples_ratio)
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
                    lost_profiles_by_dr[dr] = {"min_confidence_level" : min_confidence_level, "lost_profiles" : temp}
                last_profiles = profiles
                last_dr_values = dr_values
                profiles_by_dr[dr] = dr_values
            else : 
                profiles_by_dr[dr] = last_dr_values
        return profiles_by_dr, lost_profiles_by_dr
    
    @staticmethod
    def _filter_by_profile(X, Y_true, predicted_prob, Y_pred, mpc_values, features, path):
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
    def metrics_by_profile(Y_true, predicted_prob, Y_pred, predicted_accuracies, profile, total_population, node_population, metrics_list):
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
    Represents a profile containing metrics and values associated with a specific node.

    Attributes:
        node_id (int): An identifier for the node.
        path (str): The path (conditions) leading to the node in the tree.
        mean_value (float): The average uncertainty value of the node.
        metrics (dict, optional): Additional metrics.
    """

    def __init__(self, node_id, path, mean_value, metrics=None):
        """
        Initializes a new instance of the Profile class.
        """
        self.node_id = node_id
        self.path = path
        self.mean_value = mean_value
        self.metrics = metrics

    def to_dict(self):
        """
        Converts the Profile instance into a dictionary format suitable for serialization.

        Returns:
            dict: A dictionary representation of the Profile instance including the node ID, path, mean value, and metrics.
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
    Manages the records of profiles and lost profiles based on declaration rates and minimal samples ratio, allowing for insertion,
    retrieval, and transformation of profile data, as well as the extraction of the profiles from a computed tree representation and the
    calculation of the metrics

    Attributes:
        profiles_records (dict): A nested dictionary storing profiles organized by sample ratio and dr values.
        lost_profiles_records (dict): A nested dictionary storing lost profiles organized similarly to profiles_records.
        features (list): A list of features used in the profiles.
    """
    def __init__(self, features) -> None:
        """
        Initializes the ProfilesManager with the specified features.

        Args:
            features (list): A list of features that will be used within the profiles.
        """
        self.profiles_records = {}
        self.lost_profiles_records ={}
        self.features = features
    
    def insert_profiles(self, dr, min_samples_ratio, profiles : list):
        """
        Inserts profiles into the records under a specific dr value and minimum sample ratio.

        Args:
            dr (int): Decision rate value.
            min_samples_ratio (float): Minimum samples ratio.
            profiles (list): A list of profiles to be stored.
        """
        if min_samples_ratio not in self.profiles_records:
            self.profiles_records[min_samples_ratio] = {}
        self.profiles_records[min_samples_ratio][dr] = profiles
    
    def insert_lost_profiles(self, dr, min_samples_ratio, profiles : list):
        """
        Inserts lost profiles into the records under a specific dr value and minimum sample ratio.

        Args:
            dr (int): Decision rate value.
            min_samples_ratio (float): Minimum samples ratio.
            profiles (list): A list of lost profiles to be stored.
        """
        if min_samples_ratio not in self.lost_profiles_records:
            self.lost_profiles_records[min_samples_ratio] = {}
        self.lost_profiles_records[min_samples_ratio][dr] = profiles

    def get_profiles(self, min_samples_ratio =None, dr=None):
        """
        Retrieves profiles based on the specified minimum sample ratio and dr value.

        Args:
            min_samples_ratio (float, optional): Minimum samples ratio.
            dr (int, optional): Decision rate value.

        Returns:
            list: Profiles corresponding to the specified filters.
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
        
    def get_lost_profiles(self, min_samples_ratio =None, dr=None):
        """
        Retrieves profiles based on the specified minimum sample ratio and dr value.

        Args:
            min_samples_ratio (float, optional): Minimum samples ratio.
            dr (int, optional): Decision rate value.

        Returns:
            list: Profiles corresponding to the specified filters.
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
        Transforms a list of profile data into instances of the Profile class or dictionaries,
        depending on the `to_dict` flag.

        Args:
            profiles_list (list): A list of dictionaries, each containing keys 'id', 'path', and 'value' representing profile data.
            to_dict (bool, optional): A flag to determine if the profile instances should be returned as dictionaries. 
                                    If True, returns dictionaries; otherwise, returns Profile instances. Defaults to True.

        Returns:
            list: A list of Profile instances or dictionaries formatted from the input data, based on the `to_dict` parameter.
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
        Calculates profiles for different decision rates (dr) and minimum sample ratios by
        assessing changes in profiles across confidence levels derived from predicted accuracies.

        This method iteratively assesses profiles from a given TreeRepresentation across a range
        of minimum sample ratios and decision rates, identifying and recording both current and lost
        profiles based on the changing decision rate threshold.

        Args:
            tree (TreeRepresentation): A tree structure from which profiles are generated.
            predicted_accuracies (np.ndarray): An array of predicted accuracy values to sort and use for thresholding profiles.
            ratio_start (int): The starting point for the range of minimum sample ratios.
            ratio_end (int): The end point for the range of minimum sample ratios (exclusive).
            ratio_step (int): The increment between each sample ratio in the range.

        Modifies:
            self.profiles_records: Updates with new profiles for each combination of decision rate and sample ratio.
            self.lost_profiles_records: Updates with lost profiles for the same combinations.
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
                        lost_profiles_by_dr[dr] = {"min_confidence_level": min_confidence_level, "lost_profiles": lost_profiles}
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
        Filters datasets based on a given path of conditions derived from a profile. This method
        allows for the selection of subsets of data corresponding to specific criteria defined in the path.

        Args:
            X (np.ndarray): The feature dataset from which rows are to be selected.
            Y_true (np.ndarray): The true labels corresponding to the dataset X.
            predicted_prob (np.ndarray): The predicted probabilities corresponding to the dataset X.
            Y_pred (np.ndarray): The predicted labels corresponding to the dataset X.
            mpc_values (np.ndarray): The mpc values corresponding to the dataset X.
            path (list): A list of conditions defining the path to filter by, with each condition formatted as "column_name operator value".

        Returns:
            tuple: Contains filtered versions of X, Y_true, predicted_prob, Y_pred, and mpc_values based on the path conditions.

        Raises:
            ValueError: If an unsupported operator is included in any condition.
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

    def calc_metrics_by_profiles(self, all_x, all_y_true, all_pred_prob, all_y_pred, all_mpc_values, metrics_list):
        """
        Calculates various metrics for different profiles and decision rates based on the provided datasets.
        This method filters the dataset for each profile according to specific conditions, evaluates given metrics,
        and appends these metrics to each profile.

        Args:
            all_x (np.ndarray): The complete set of features for all observations.
            all_y_true (np.ndarray): The true labels for all observations.
            all_pred_prob (np.ndarray): The predicted probabilities for each observation.
            all_y_pred (np.ndarray): The predicted labels for each observation.
            all_mpc_values (np.ndarray): The model prediction confidence values for all observations.
            metrics_list (list): A list of metric objects that implement a calculate method for evaluating the dataset.

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
                    x, y_true, pred_prob, y_pred, mpc_values = self._filter_by_profile(all_x, all_y_true, all_pred_prob, all_y_pred, all_mpc_values, profile['path'])
                    metrics_dict = {}
                    for metric in metrics_list:
                        if isinstance(metric, RocAuc):
                            metrics_dict['RocAuc'] = metric.calculate(y_true[mpc_values >= min_confidence_level], pred_prob[mpc_values >= min_confidence_level])
                        elif isinstance(metric, AveragePrecision):
                            metrics_dict['AveragePrecision'] = metric.calculate(y_true[mpc_values >= min_confidence_level], pred_prob[mpc_values >= min_confidence_level])
                        else:
                            metrics_dict[metric.__class__.__name__] = metric.calculate(y_true[mpc_values >= min_confidence_level], y_pred[mpc_values >= min_confidence_level])
                    
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



'''

# Feature names
features = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp']

# X data (20 samples, 5 features each)
X_samples = np.array([
    [25, 175, 80, 120, 80],
    [30, 180, 85, 126, 82],
    [35, 165, 70, 130, 88],
    [40, 160, 60, 132, 85],
    [45, 170, 75, 140, 90],
    [50, 175, 80, 138, 89],
    [55, 165, 85, 145, 92],
    [60, 170, 70, 150, 94],
    [65, 160, 65, 155, 93],
    [70, 175, 75, 160, 95],
    [75, 180, 80, 165, 97],
    [80, 165, 85, 170, 96],
    [85, 170, 90, 175, 98],
    [22, 172, 67, 110, 70],
    [28, 168, 70, 115, 75],
    [32, 174, 73, 118, 76],
    [36, 169, 75, 122, 78],
    [41, 165, 77, 125, 79],
    [45, 163, 72, 128, 77],
    [52, 170, 80, 130, 85],
])

# Y data (binary outcomes for the 20 samples)
Y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

predicted_prob = np.array([
    0.9, 0.75, 0.6, 0.4, 0.85, 
    0.2, 0.95, 0.1, 0.5, 0.8, 
    0.55, 0.3, 0.65, 0.45, 0.7, 
    0.25, 0.15, 0.35, 0.05, 0.98
])

# step 1 : calculate the uncertainty or the confidence level of a base model
uncertainty_calc = UncertaintyCalculator(AbsoluteError)
uncertainty_values = uncertainty_calc.calculate_uncertainty(X_samples, predicted_prob, Y_true)

# step 2 : Calculate the predictions based on a threshold (here 0.5) and sample weight (here 1 for all)
y_pred = np.array([1 if y_score_i >= 0.5 else 0 for y_score_i in predicted_prob])
sample_weight = np.full(20, 1)

# step 3 : Create and train the IPCModel
IPC_model = IPCModel()
param_grid = {
        'max_depth': range(2, 4)
    }
IPC_model.optimize(param_grid, 4, X_samples, uncertainty_values, sample_weight)

# step 4 : Predict IPC_values using the IPCModel
IPC_values = IPC_model.predict(X_samples)
print("Calculated confidence level :", uncertainty_values)
print("Predicted confidence level by IPC Model :", IPC_values)

# step 5 : Create and train the APCModel on IPC_values
APC_model = APCModel(features, max_depth=2)
APC_model.train(X_samples,IPC_values)

# step 6 : Predict APC_values using the APCModel
APC_values = APC_model.predict(X_samples)
print("Predicted confidence level by APC Model :", APC_values)
profiles = APC_model.treeRepresentation.get_all_profiles()

# step 7 : Create and predict the minimum confidence levels using the MPCModel
MPC_model =MPCModel(IPC_values, APC_values)
MPC_values = MPC_model.predict()
print("Predicted confidence level by MPC Model :", MPC_values)

# step 8: Calculate metrics by declaration rate
# Initialize MDRCalculator with a subset of metrics
metrics = [Accuracy(), RocAuc(), Precision(), Recall(), F1Score(), MatthewsCorrCoef(), Specificity(), Sensitivity(), BalancedAccuracy(), NPV(), PPV()]
# Calculate metrics by DR
# metrics_by_dr = MDRCalculator.calc_metrics_by_dr(Y_true, y_pred, predicted_prob, MPC_values, metrics_list=metrics)
# Print the results
# pprint(metrics_by_dr)

# step 9: Calculate profiles by declaration rate, and the lost profiles
tree = APC_model.treeRepresentation
profiles_by_dr, lost_profiles_by_dr = MDRCalculator.calc_profiles_by_dr(tree, predicted_accuracies=MPC_values, min_samples_ratio=0)
# Print the results
#pprint(profiles_by_dr)
#pprint(lost_profiles_by_dr)


prof_manager = ProfilesManager(features)
prof_manager.calc_profiles(tree, MPC_values, 0, 10, 2)
pprint(prof_manager.profiles_records)
returned_prof = prof_manager.get_profiles(4, 40)
#pprint(returned_prof)
returned_prof = prof_manager.get_lost_profiles(4, 40)
#pprint(returned_prof)
prof_manager.calc_metrics_by_profiles(X_samples, Y_true, predicted_prob, y_pred, MPC_values, metrics)
pprint(prof_manager.profiles_records)




'''

'''
filtered_x, filtered_y_true, filtered_probs, filtered_y_pred, filtered_acc = MDRCalculator._filter_by_profile(X_samples, Y_true, predicted_prob, y_pred, MPC_values, features, ['*', 'diastolic_bp <= 86.5', 'age <= 34.0'])
pprint(filtered_x)
pprint(filtered_y_true)

'''





