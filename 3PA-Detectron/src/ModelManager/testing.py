import ModelFactories
import Models
import BaseModel
import xgboost as xgb
import pickle
import numpy as np
from xgboost.sklearn import XGBClassifier

"""loaded_model = xgboost.Booster()
loaded_model.load_model('uci_heart_0.model')
pickle.dump(loaded_model, open('pickled_model.pkl','wb'))"""

"""fact = ModelFactories.ModelFactory()
model = fact.create_model_from_pickled("pickled_model.pkl")"""

"""model = pickle.load(open('pickled_model.pkl', 'rb'))
print(model)
print(type(model))

if isinstance(model, xgb.Booster):
    print('true')
else : 
    print('no')"""
"""
print("Model with params exemple")
params = {'objective': 'binary:logitraw'}
fact = ModelFactories.ModelFactory()
model = fact.create_model_with_hyperparams("XGBoostModel", params)
print(model.model_class)
print(model.pickled_model)
print(model.params)

print("Model with pickled file exemple")
fact = ModelFactories.ModelFactory()
model = fact.create_model_from_pickled('pickled_model.pkl')
print(model.model_class)
print(model.pickled_model)
print(model.params)


print("BaseModel setting exemple")
fact = ModelFactories.ModelFactory()
model = fact.create_model_from_pickled('pickled_model.pkl')
# print(Models.BaseModelManager.get_instance())
BaseModel.BaseModelManager.set_base_model(model)
print(BaseModel.BaseModelManager.get_instance().model)
# cloned_model = type(BaseModel.BaseModelManager.get_instance())()
# print(type(cloned_model))
# Models.BaseModelManager.set_base_model(model)
cloned_model = BaseModel.BaseModelManager.clone_base_model()
print(cloned_model.model)

"""

# Prepare training data
features = np.array([[1, 2], [3, 4]])
labels = np.array([1, 0])
dtrain = xgb.DMatrix(features, label=labels)

# Train xgb.Booster
params = {'objective': 'binary:logistic', 'use_label_encoder': False}
num_rounds = 10
booster_model = xgb.train(params, dtrain, num_rounds)

# Pickle the xgb.Booster model
with open('booster_model.pkl', 'wb') as file:
    pickle.dump(booster_model, file)

# Train xgb.XGBClassifier
classifier_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
classifier_model.fit(features, labels)

# Pickle the xgb.XGBClassifier model
with open('classifier_model.pkl', 'wb') as file:
    pickle.dump(classifier_model, file)