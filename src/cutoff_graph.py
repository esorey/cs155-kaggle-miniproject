import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, feature_selection
from matplotlib import pyplot as plt
from sklearn.externals import joblib

#for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
infile = 'data/train_2008.csv'
data = genfromtxt(infile, delimiter=',')
print("Data successfully loaded from %s" % infile + '...')

# Trim the first row which contains header information
# and the first three columns which contain irrelevant data
data = data[1:, 3:]
x_train = data[:, 0:-1]
y_train = data[:, -1]
partition = 55000
x_test = x_train[partition:, :]
y_test = y_train[partition:]
x_train = x_train[0:partition, :]
y_train = y_train[0:partition]


# Scale data to have 0 mean and unit variance
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Train Gradient Boosting Classifier
print("Begin training Gradient Boosting...")
rf = ensemble.GradientBoostingClassifier(n_estimators=500, min_samples_leaf=15)
rf.fit(x_train, y_train)

trng_acc = rf.score(x_train, y_train)
val_acc = rf.score(x_test, y_test)
#print("Threshold: %f" %thresh)
print("Training accuracy: %f" %trng_acc)
print("Validation accuracy: %f" %val_acc)
