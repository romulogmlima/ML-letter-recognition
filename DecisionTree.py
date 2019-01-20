# author: @romulogmlima

import pandas
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


url = "letter-recog.csv"

names = ['class',
         'x-box',
         'y-box',
         'width',
         'high',
         'onpix',
         'x-bar',
         'y-bar',
         'x2bar',
         'y2bar',
         'xybar',
         'x2ybr',
         'xy2br',
         'x-ege',
         'xegvy',
         'y-ege',
         'yegvx']

dataset = pandas.read_csv(url, names=names)

# shape
# print(dataset.shape)

# Generate various summary statistics
# print(dataset.describe())

# class distribution
# print(dataset.groupby('class').size())

# histograms
# dataset.hist()
# plt.show()

array = dataset.values

X = array[:, 1:17]
Y = array[:, 0]

print('X matrix dimensionality:', X.shape)
print('Y vector dimensionality:', Y.shape)

# split the data into a training set and a test set
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20,
                                                                                random_state=10, stratify=Y)

# print("X_train: ", X_train.shape)
# print("X_validation: ", X_validation.shape))
# print("Y_train: ", Y_train.shape))
# print("Y_validation: ", Y_validation.shape))


dec_tree = DecisionTreeClassifier()

cm = ConfusionMatrix(dec_tree, classes="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(','))

# train the model using the training sets
cm.fit(X_train, Y_train)

cm.score(X_validation, Y_validation)

# predict the responses for test dataset
predictions = cm.predict(X_validation)

# accuracy classification score
print("Accuracy: ", accuracy_score(Y_validation, predictions))

# compute confusion matrix
print(confusion_matrix(Y_validation, predictions))

# text report showing the main classification metrics
print(classification_report(Y_validation, predictions, digits=5))

cm.poof()
