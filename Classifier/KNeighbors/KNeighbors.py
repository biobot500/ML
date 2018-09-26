X = [[0,0,1], [1,0,0], [0,1,0], [1,1,0]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 

print(neigh.predict([[0,0,1]]))

#print(neigh.predict_proba([[0.9]]))