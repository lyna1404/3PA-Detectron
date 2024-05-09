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
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'hist',
    'device': 'cpu'
}


loaded_model = ModelFactory.create_model_from_pickled("./src/Detectron2/model.pkl")

datasets = DatasetsManager()
datasets.set_baseModel_training_data("./src/Detectron2/bm_train_dataset.csv", "y_true")
datasets.set_baseModel_validation_data("./src/Detectron2/bm_val_dataset.csv", "y_true")
datasets.set_reference_data("./src/Detectron2/bm_test_dataset.csv", "y_true")
datasets.set_testing_data("./src/Detectron2/ood_sampled.csv", "y_true")
train_x, train_y = datasets.get_base_model_training_data()


datasets2 = DatasetsManager()
datasets2.set_baseModel_training_data("./src/Detectron2/bm_train_dataset.csv", "y_true")
datasets2.set_baseModel_validation_data("./src/Detectron2/bm_val_dataset.csv", "y_true")
datasets2.set_reference_data("./src/Detectron2/bm_test_dataset.csv", "y_true")
datasets2.set_testing_data("./src/Detectron2/ood_hungary_sampled.csv", "y_true")


bm_manager = BaseModelManager()
bm_manager.set_base_model(loaded_model)

experiment = DetectronExperiment()

print("///////////////////////// Base model evaluation on the different datasets : ////////////////////////////////////////")
x_test_cleveland, y_test_cleveland = datasets.get_reference_data()
x_test_va, y_test_va = datasets.get_testing_data()
x_test_hungary, y_test_hungary = datasets2.get_testing_data()
print("Base model evaluation on Cleveland")
loaded_model.evaluate(x_test_cleveland, y_test_cleveland, ['Auc', 'Accuracy'], True)
print("Base model evaluation on VA Long Beach")
loaded_model.evaluate(x_test_va, y_test_va, ['Auc', 'Accuracy'], True)
print("Base model evaluation on Hungary")
loaded_model.evaluate(x_test_hungary, y_test_hungary, ['Auc', 'Accuracy'], True)



print("///////////////////////// Detectron experiment with different tests : ////////////////////////////////////////")
print("///////////////////////// Va Long Beach : ////////////////////////////////////////")

print("Running VA Long Beach using Tom test")
exp_res, eval_res = experiment.run(datasets=datasets, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy())
print(exp_res)
print(eval_res)

print("Running VA Long Beach using quantile test")
exp_res, eval_res = experiment.run(datasets=datasets, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_quantile())
print(exp_res)
print(eval_res)


print("Running VA Long Beach using Mann Whitney u test and 100 test runs")
exp_res, eval_res = experiment.run(datasets=datasets, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_MW())
print(exp_res)

print("Running VA Long Beach using ks test and 100 test runs")
exp_res, eval_res = experiment.run(datasets=datasets, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_KS())
print(exp_res)

print("///////////////////////// Hungary : ////////////////////////////////////////")

print("Running Hungary using Tom test")
exp_res, eval_res = experiment.run(datasets=datasets2, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy(), evaluate_detectron=True)
print(exp_res)

print("Running Hungary using quantile test")
exp_res, eval_res = experiment.run(datasets=datasets2, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_quantile())
print(exp_res)

print("Running Hungary using Mann Whitney u test and 100 test runs")
exp_res, eval_res = experiment.run(datasets=datasets2, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_MW())
print(exp_res)

print("Running Hungary using ks test and 100 test runs")
exp_res, eval_res = experiment.run(datasets=datasets2, training_params=XGB_PARAMS, base_model_manager=bm_manager, test_strategy=DisagreementStrategy_KS())
print(exp_res)

