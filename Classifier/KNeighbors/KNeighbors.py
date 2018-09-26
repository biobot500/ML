from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import pandas as pd

DATA_PATH = './data/'
def GetVector(img,FromData=True):
    if FromData==True:
        i = cv2.imread(DATA_PATH+img)
    else:
        i = cv2.imread(img)

    i = cv2.resize(i, (224, 224))
    array = np.array(i)
    vector = array.reshape(-1)
    return vector


df = pd.read_csv('data.csv');
images = df[['image_url']]

X = []


for index,image in images.iterrows():

        vector = GetVector(image[0])
        X.append(vector)

y = df.name;

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)

file_name = 'test.jpg'


proba = neigh.predict_proba([GetVector(file_name,False)])[0];
#print(proba)
#print(neigh.predict([GetVector(file_name, False)]))
found = False
for score in proba:

    if score >= 0.9:
        found = True
        #print(score)

if found == False:
    print "NO MATCH FOUND"
else:
    print(neigh.predict([GetVector(file_name, False)]))

'''
if neigh.predict_proba([GetVector(file_name,False)])[0][0] >= 1.0:
    print(neigh.predict([GetVector(file_name,False)]))
else:
    print "No Match FOund"
'''
