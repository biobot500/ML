import pickle

Model = pickle.load(open('Models/RideFraudModel','rb'))
print Model.predict([[10,20,25]])