from sklearn.naive_bayes import MultinomialNB
import numpy as np
X = [
        [
            1,32,12,32
        ],
        [
            10, 0, 10, 32
        ],
        [
            1, 0, 0 ,0
        ]
    ]

y = np.array([1, 2, 3])

clf = MultinomialNB()
clf.fit(X, y)



print clf.predict([[11,2,11,40]])

