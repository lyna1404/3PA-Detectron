# required imports
import torch
import seaborn as sns
import numpy as np
import xgboost as xgb
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from ..DatasetManager.Datasets import DatasetsManager
from ..ModelManager.ModelFactories import ModelFactory, XGBoostFactory
import pickle


'''
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'custom_eval_metrics': ['Auc', 'Accuracy'],
    'eta': 0.001,
    'max_depth':4,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    'min_child_weight': 4,
    'nthread': 4,
    'tree_method': 'hist',
    'device': 'cpu',
    'eval_metric':'auc',
    'num_boost_rounds':50,
}

datasets = DatasetsManager()
datasets.set_baseModel_training_data("./src/Detectron2/diabetes_train_data.csv", 'Outcome')
datasets.set_baseModel_validation_data('./src/Detectron2/diabetes_validation_data.csv', 'Outcome')
datasets.set_reference_data('./src/Detectron2/diabetes_test_data.csv', 'Outcome')
datasets.set_testing_data('./src/Detectron2/diabetes_test_scaled_data.csv', 'Outcome')

factory =ModelFactory()
loaded_model = factory.create_model_from_pickled('./diabetes_xgb_model.pkl')
x_train, y_train = datasets.get_base_model_training_data()
x_val, y_val = datasets.get_base_model_validation_data()
x_test, y_test = datasets.get_reference_data()
x_test_ood, y_test_ood = datasets.get_testing_data()


loaded_model.evaluate(x_test, y_test, ['Auc', 'Accuracy'], True)
loaded_model.evaluate(x_test_ood, y_test_ood, ['Auc', 'Accuracy'], True)






# Define a parameter grid to search over
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 4, 6],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'eta': [0.01, 0.05, 0.1],
}

# Create a XGBClassifier object
xgb_model = XGBClassifier(objective='binary:logistic', n_estimators=100, nthread=4, eval_metric='auc', use_label_encoder=False)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best AUC: ", grid_search.best_score_)

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)'''



# Load the dataset
file_path = './src/Detectron2/diabetes_test_data.csv'  # Replace this with the actual path to your CSV file
data = pd.read_csv(file_path)

# Select columns to add noise
columns_to_noise = ['Pregnancies', 'Age', 'Insulin', 'DiabetesPedigreeFunction', 'Glucose']  # Replace with actual column names

# Apply Gaussian noise to each selected column
for column in columns_to_noise:
    if column in data.columns:
        # Analyze column to set appropriate noise level
        col_std = data[column].std()

        # Define noise parameters (e.g., 10% of the column's standard deviation)
        noise = np.random.normal(loc=0, scale=col_std * 0.9, size=data[column].shape)

        # Add noise to the column
        data[column] += noise

scaling_factor = 3.4
for feature in columns_to_noise:
    if feature in data.columns:
        data[feature] *= scaling_factor
        col_std = data[feature].std()

        # Define noise parameters (e.g., 10% of the column's standard deviation)
        noise = np.random.normal(loc=0, scale=col_std * 0.9, size=data[feature].shape)

        # Add noise to the column
        data[feature] += noise

# Save the modified dataset
output_file_path = './src/Detectron2/diabetes_test_scaled_data.csv'
data.to_csv(output_file_path, index=False)

