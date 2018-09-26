from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np

def GetVector(img):
    i = cv2.imread(img)
    i = cv2.resize(i, (224, 224))
    array = np.array(i)


    return array.reshape(-1)




#print GetVector('im1.jpg')

X = [GetVector('im1.jpg'),GetVector('im2.jpg')]
y = [0, 1]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y) 

print(neigh.predict([GetVector('test.jpeg')]))

#print(neigh.predict_proba([[0.9]]))
