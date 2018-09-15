import pandas as pd
from sklearn import linear_model

df = pd.read_csv("data.csv")

linReg = linear_model.LinearRegression();
linReg.fit(df[['area','bedrooms','age']],df.price)

print linReg.predict([[1200,3,10]])