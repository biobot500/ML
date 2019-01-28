from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model

X = [
        [
            # through panthapath to bashundhara shopping
            23.752552, 90.381469, 23.752755, 90.383015, 23.752552, 90.384639, 23.751556, 90.386185, 23.750807, 90.388417, 23.750669, 90.390391
        ],
        [
            # through green road to bashundhara shopping
            23.749020, 90.381786, 23.748823, 90.383610, 23.748902, 90.385112, 23.749078, 90.386163, 23.750669, 90.387000, 23.750728, 90.390219
        ],
        [
            # through garden road to bashundhara shopping
            23.752457, 90.381099, 23.752771, 90.384254, 23.751632, 90.386142, 23.751337, 90.387193, 23.752555, 90.387837, 23.752496, 90.389726
        ]
    ]

y = ["route_1", "route_2", "route_3"]

clf = MultinomialNB()
clf.fit(X, y)


input = [
            [23.750938, 90.381639,23.748660, 90.381896,23.748807, 90.384396, 23.749337, 90.386467,23.751085, 90.387239, 23.750653, 90.390340]
        ]
travelRoute = clf.predict(input)[0]



linReg = linear_model.LinearRegression();



print(travelRoute)