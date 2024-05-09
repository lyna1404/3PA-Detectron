from Datasets import DatasetsManager

manager = DatasetsManager()
manager.set_baseModel_training("simulated_data.csv", "y_true")
print(manager.base_model_training_set.features)
print(manager.base_model_training_set.true_labels)

