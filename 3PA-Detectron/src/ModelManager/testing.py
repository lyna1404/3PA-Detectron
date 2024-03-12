import ModelFactories
import Models
import xgboost as xgb
import pickle

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