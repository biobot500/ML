from sklearn.neighbors import KNeighborsClassifier
from GetVector import GetVector
import pandas as pd
import pickle

df = pd.read_csv('data.csv');
images = df[['image_url']]

X = []


for index,image in images.iterrows():

        vector = GetVector(image[0])
        X.append(vector)

y = df.name;

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
pickle.dump(neigh,open('model/ImageClassifier','wb'));

print "TRAINING COMPLETE";

