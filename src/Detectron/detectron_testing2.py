from .experiment import DetectronExperiment
from ..DatasetManager.Datasets import DatasetsManager
from ..ModelManager.BaseModel import BaseModelManager
from ..ModelManager.ModelFactories import ModelFactory
from .strategies import DisagreementStrategy, DisagreementStrategy_MW, DisagreementStrategy_KS, DisagreementStrategy_quantile
import torch 
from pprint import pprint
import pandas as pd
import numpy as np
import pickle
import json

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



factory =ModelFactory()
loaded_model = factory.create_model_from_pickled('./diabetes_xgb_model.pkl')

datasets = DatasetsManager()
datasets.set_baseModel_training_data("./src/Detectron2/diabetes_train_data.csv", 'Outcome')
datasets.set_baseModel_validation_data('./src/Detectron2/diabetes_validation_data.csv', 'Outcome')
datasets.set_reference_data('./src/Detectron2/diabetes_test_data_1_half.csv', 'Outcome')
datasets.set_testing_data('./src/Detectron2/diabetes_test_data_2_half.csv', 'Outcome')

x_train, y_train = datasets.get_base_model_training_data()
x_val, y_val = datasets.get_base_model_validation_data()
x_test, y_test = datasets.get_reference_data()
x_test_ood, y_test_ood = datasets.get_testing_data()

bm_manager = BaseModelManager()
bm_manager.set_base_model(loaded_model)

experiment = DetectronExperiment()

print("///////////////////////// Base model evaluation on the different datasets : ////////////////////////////////////////")

print("Base model evaluation on iid test data")
loaded_model.evaluate(x_test, y_test, ['Auc', 'Accuracy'], True)
print("Base model evaluation on ood test data")
loaded_model.evaluate(x_test_ood, y_test_ood, ['Auc', 'Accuracy'], True)




print("///////////////////////// Detectron experiment with different tests : ////////////////////////////////////////")

print("Running ood with Tom test")
detectron_results, exp_res, eval_res = experiment.run(datasets=datasets, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy(), allow_margin=True, margin=0.03)
print(exp_res)
print(eval_res)
print(detectron_results)
print("Running ood using quantile test")
_, exp_res, eval_res = experiment.run(datasets=datasets, detectron_result=detectron_results, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_quantile(),  allow_margin=True, margin=0.03)
print(exp_res)
print(eval_res)


print("Running ood using Mann Whitney u test and 100 test runs")
_, exp_res, eval_res = experiment.run(datasets=datasets, detectron_result=detectron_results, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_MW(), allow_margin=True, margin=0.03)
print(exp_res)

print("Running ood using ks test and 100 test runs")
_, exp_res, eval_res = experiment.run(datasets=datasets, detectron_result=detectron_results, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_KS(), allow_margin=True, margin=0.03)
print(exp_res)

'''

datasets2 = DatasetsManager()
datasets2.set_baseModel_training_data("./src/Detectron2/diabetes_train_data.csv", 'Outcome')
datasets2.set_baseModel_validation_data('./src/Detectron2/diabetes_validation_data.csv', 'Outcome')
datasets2.set_reference_data('./src/Detectron2/diabetes_test_data.csv', 'Outcome')
datasets2.set_testing_data('./src/Detectron2/diabetes_modified_0.1_test_data.csv', 'Outcome')

x_test, y_test = datasets2.get_reference_data()
x_test_ood, y_test_ood = datasets2.get_testing_data()

print("///////////////////////// Base model evaluation on the different datasets : ////////////////////////////////////////")

print("Base model evaluation on iid test data")
loaded_model.evaluate(x_test, y_test, ['Auc', 'Accuracy'], True)
print("Base model evaluation on ood test data")
loaded_model.evaluate(x_test_ood, y_test_ood, ['Auc', 'Accuracy'], True)

print("///////////////////////// Detectron experiment with different tests : ////////////////////////////////////////")

print("Running ood with Tom test")
detectron_results2, exp_res, eval_res = experiment.run(datasets=datasets2, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy(),  allow_margin=True, margin=0.3)
print(exp_res)
print(eval_res)

print("Running ood using quantile test")
_, exp_res, eval_res = experiment.run(datasets=datasets2, detectron_result=detectron_results2 ,training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_quantile(), allow_margin=True, margin=0.03)
print(exp_res)
print(eval_res)


print("Running ood using Mann Whitney u test and 100 test runs")
_, exp_res, eval_res = experiment.run(datasets=datasets2, detectron_result=detectron_results2 , training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_MW(), allow_margin=True, margin=0.03)
print(exp_res)

print("Running ood using ks test and 100 test runs")
_, exp_res, eval_res = experiment.run(datasets=datasets2, detectron_result=detectron_results2 , training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_KS(),  allow_margin=True, margin=0.03)
print(exp_res)



df = pd.read_csv('./src/Detectron2/diabetes_test_data.csv')

# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# Split the DataFrame into two equal parts
half = int(len(df) / 2)
first_half = df[:half]
second_half = df[half:]

# Optionally, save the split dataframes back to new CSV files
first_half.to_csv('./src/Detectron2/diabetes_test_data_1_half.csv', index=False)
second_half.to_csv('./src/Detectron2/diabetes_test_data_2_half.csv', index=False)
'''

