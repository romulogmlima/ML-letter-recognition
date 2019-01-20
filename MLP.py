# author: @romulogmlima

import pandas
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


url = "letter-recog.csv"

names = ['Class',
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

X = dataset.iloc[:, 1:17]
Y = dataset.select_dtypes(include=[object])

#Y.Class.unique()

#le = preprocessing.LabelEncoder()

#Y = Y.apply(le.fit_transform)


# split the data into a training set and a test set
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20,
                                                                                random_state=10)

# print("X_train: ", X_train.shape)
# print("X_validation: ", X_validation.shape))
# print("Y_train: ", Y_train.shape))
# print("Y_validation: ", Y_validation.shape))


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)


mlp = MLPClassifier(hidden_layer_sizes=(250, 300), max_iter=1000000, activation='logistic')

cm = ConfusionMatrix(mlp, classes="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(','))

# train the model using the training sets
cm.fit(X_train, Y_train.values.ravel())

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
