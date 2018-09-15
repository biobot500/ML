import pandas as pd
from sklearn import linear_model

df = pd.read_csv("data.csv")

area = df[['area']]
price = df.price

reg = linear_model.LinearRegression()
reg.fit(area,price)

print reg.predict(7000)
