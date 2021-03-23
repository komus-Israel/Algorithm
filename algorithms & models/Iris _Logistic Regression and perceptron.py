from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data =  datasets.load_iris()

features = data.data

target = data.target


X = features[:100,]
y = target[:100]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)

from perceptron import Perceptron

clf = Perceptron()

train = clf.fit(X_train, y_train)

predict = clf.predict(X_test)

accuracy =  accuracy_score(predict, y_test)

print('perceptron\'s Prediction \n', predict)

print('The accuracy of perceptron prediction is',accuracy)


print()
from LogisticRegression import LogisticRegression

lg = LogisticRegression(eta = 0.01)

model = lg.fit(X_train, y_train)

predict2 = lg.predict(X_test)

accuracy2 =  accuracy_score(predict2, y_test)

print('The accuracy of Logistic Regression\'s prediction is',accuracy2)

print('logistic Regression Prediction \n', predict2)

print('y_test \n', y_test)

print('The estimated sigmoid function of the predicted values are \n',lg.estimate(X_test))


