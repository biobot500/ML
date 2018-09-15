import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
area = df[['area']]
price = df.price

reg = linear_model.LinearRegression()
reg.fit(area,price)
print reg.predict(7000)
