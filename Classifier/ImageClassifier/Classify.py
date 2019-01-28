from flask import Flask
from flask import request
from GetVector import GetVector
import urllib
import pickle


app = Flask(__name__)

@app.route("/classify-image")
def classify_image():
    img_url = request.args.get('url', '')
    urllib.urlretrieve(img_url, "test.jpg")
    file="test.jpg"
    neigh = pickle.load(open('model/ImageClassifier','rb'))
    print neigh.predict_proba([GetVector(file, False)])
    return neigh.predict([GetVector(file, False)])[0]

    '''
    proba = neigh.predict_proba([GetVector(file, False)])[0];
    found = False
    for score in proba:
        print(score)
        if score >= 0.9:
            found = True
            print(score)

    if found == False:
        return "NO MATCH FOUND"
    else:
        return neigh.predict([GetVector(file, False)])[0]
    '''


