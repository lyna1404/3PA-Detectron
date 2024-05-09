import numpy as np
import pandas as pd
from ..ModelManager.eval_metrics import EvaluationMetric, Accuracy, AveragePrecision, Recall, RocAuc, F1Score, MatthewsCorrCoef, Precision, Sensitivity, Specificity, BalancedAccuracy, PPV, NPV
from pprint import pprint
from .tree_structure import TreeRepresentation
from .Models import IPCModel, APCModel, MPCModel
from .uncertainty import UncertaintyCalculator, AbsoluteError
from .metrics import ProfilesManager, MDRCalculator
import json

# Tolerance for floating-point comparisons
tolerance = 1e-8 

# Load data from CSV file
data = pd.read_csv('./src/Med3pa/simulated_data.csv')

# Extract feature matrix and labels
X_samples = data[['x1', 'x2']].values
Y_true = data['y_true'].values
predicted_prob = data['pred_prob'].values
features = ['x1', 'x2']

# step 1 : calculate the uncertainty or the confidence level of a base model
uncertainty_calc = UncertaintyCalculator(AbsoluteError)
uncertainty_values = uncertainty_calc.calculate_uncertainty(X_samples, predicted_prob, Y_true)

# step 2 : Calculate the predictions based on a threshold (here 0.5) and sample weight (here 1 for all)
y_pred = np.array([1 if y_score_i >= 0.5 else 0 for y_score_i in predicted_prob])
sample_weight = np.full(X_samples.shape[0], 1)

# step 3 : Create and train the IPCModel
max_depth_log = int(np.log2(X_samples.shape[0]))

IPC_model = IPCModel()
param_grid = {
    'max_depth': range(2, max_depth_log + 1)
}
IPC_model.optimize(param_grid, min(4, int(X_samples.shape[0] / 2)), X_samples, uncertainty_values, sample_weight)

# step 4 : Predict IPC_values using the IPCModel
IPC_values = IPC_model.predict(X_samples)
# loading olivier's results
Olivier_IPC_values = np.load('./src/Med3pa/ca_rf_values.npy')
# Check if all elements are almost equal considering a tolerance
almost_identical = np.allclose(IPC_values, Olivier_IPC_values, atol=tolerance)
print("IPC values are almost identical:", almost_identical)
#print("Calculated confidence level :", uncertainty_values)
#print("Predicted confidence level by IPC Model :", IPC_values)

# step 5 : Create and train the APCModel on IPC_values
APC_model = APCModel(features, max_depth=3, min_sample_ratio=-5)
APC_model.train(X_samples,IPC_values)

# step 6 : Predict APC_values using the APCModel
APC_values = APC_model.predict(X_samples, min_samples_ratio=5)
# loading olivier's results
Olivier_APC_values = np.load('./src/Med3pa/ca_profile_values_5.npy')
almost_identical = np.allclose(APC_values, Olivier_APC_values, atol=tolerance)
print("APC values are almost identical:", almost_identical)

#print("Predicted confidence level by APC Model :", APC_values)
profiles = APC_model.treeRepresentation.get_all_profiles()

# step 7 : Create and predict the minimum confidence levels using the MPCModel
MPC_model =MPCModel(IPC_values, APC_values)
Olivier_MPC_values = np.load('./src/Med3pa/mpc_values_5.npy')
MPC_values = MPC_model.predict()
almost_identical = np.allclose(MPC_values, Olivier_MPC_values, atol=tolerance)
print("MPC values are almost identical:", almost_identical)

#print("Predicted confidence level by MPC Model :", MPC_values)

# step 8: Calculate metrics by declaration rate
# Initialize MDRCalculator with a subset of metrics
metrics = [Accuracy(), RocAuc(), Precision(), Recall(), F1Score(), MatthewsCorrCoef(), Specificity(), Sensitivity(), BalancedAccuracy(), NPV(), PPV()]
# Calculate metrics by DR
metrics_by_dr = MDRCalculator.calc_metrics_by_dr(Y_true, y_pred, predicted_prob, MPC_values, metrics_list=metrics)
filename = 'metrics_dr.json'

# Write JSON data to file
with open(filename, 'w') as file:
    json.dump(metrics_by_dr, file, indent=4)


# step 9: Calculate profiles by declaration rate, and the lost profiles
tree = APC_model.treeRepresentation
prof_manager = ProfilesManager(features)
prof_manager.calc_profiles(tree, MPC_values, 0, 11, 5)
# save the profiles to a json file
filename = 'profiles.json'

# Write JSON data to file
with open(filename, 'w') as file:
    json.dump(prof_manager.profiles_records, file, indent=4)

# save the lost profiles to a json file
filename = 'lost_profiles.json'

# Write JSON data to file
with open(filename, 'w') as file:
    json.dump(prof_manager.lost_profiles_records, file, indent=4)


prof_manager.calc_metrics_by_profiles(X_samples, Y_true, predicted_prob, y_pred, MPC_values, metrics)

# save the profiles metrics to a json file
filename = 'profiles_metrics.json'

# Write JSON data to file
with open(filename, 'w') as file:
    json.dump(prof_manager.profiles_records, file, indent=4)
