from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle


dataset = pd.read_csv('data.csv')
x = dataset[['total_km','total_time','speed']]
y = dataset.fraud


model = GaussianNB()
model.fit(x, y)

#save
pickle.dump(model,open('Models/RideFraudModel','wb'));


