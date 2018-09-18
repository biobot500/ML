import pickle

Model = pickle.load(open('Models/RideFraudModel','rb'))
print Model.predict([[23.751221, 90.386718,23.749433, 90.385302,23.751270, 90.383489]])