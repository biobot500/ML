from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np

def GetVector(img):
    i = cv2.imread(img)
    i = cv2.resize(i, (224, 224))
    array = np.array(i)
    vector = array.reshape(-1)
    return vector





X = [
        GetVector('data/chair1.jpeg'),
        GetVector('data/chair2.jpeg'),
        GetVector('data/chair3.jpeg'),
        GetVector('data/table.JPG'),
        GetVector('data/table2.jpg'),
        GetVector('data/table3.jpeg'),
     ]

y = ['chair', 'chair', 'chair','table','table','table']
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
print neigh.predict_proba([GetVector('chair1.jpg')])[0][0];
if neigh.predict_proba([GetVector('chair1.jpg')])[0][0] >= 1.0:
    print(neigh.predict([GetVector('chair1.jpg')]))
else:
    print "No Match FOund"

#print(neigh.predict_proba([[0.9]]))
