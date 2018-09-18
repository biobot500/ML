from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle


dataset = pd.read_csv('data.csv')
x = dataset[['point_1_lat','point_1_lng','point_2_lat','point_2_lng','point_3_lat','point_3_lng']]
y = dataset.total_km


model = GaussianNB()
model.fit(x, y)

#save
pickle.dump(model,open('Models/RideFraudModel','wb'));


