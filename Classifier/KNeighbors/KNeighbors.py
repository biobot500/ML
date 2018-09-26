from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np

def GetVector(img):
    i = cv2.imread(img)
    i = cv2.resize(i, (224, 224))
    array = np.array(i)
    vector = array.reshape(-1)
    return vector





X = [GetVector('data/chair1.jpeg'),GetVector('data/chair2.jpeg'),GetVector('data/chair3.jpeg')]

y = ['chair-1', 'chair-2', 'chair-3']
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)

if neigh.predict_proba([GetVector('chair2.jpeg')])[0][1] >= 1.0:
    print(neigh.predict([GetVector('chair2.jpeg')]))
else:
    print "No Match FOund"

#print(neigh.predict_proba([[0.9]]))
