import numpy as np
from sklearn.naive_bayes import BernoulliNB

X = [
    [1,1],
    [1,0],
    [0,1],
    [0,0],
    [0,1]
]
Y = np.array([1,2,3,4,5])

clf = BernoulliNB()
clf.fit(X, Y)

print(clf.predict([[0,0]]))