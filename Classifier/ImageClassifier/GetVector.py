import cv2
import numpy as np

def GetVector(img,FromData=True):
    DATA_PATH = './data/'
    print(img)
    if FromData==True:
        i = cv2.imread(DATA_PATH+img)
    else:
        i = cv2.imread(img)

    i = cv2.resize(i, (100, 100))
    array = np.array(i)
    vector = array.reshape(-1)
    return vector