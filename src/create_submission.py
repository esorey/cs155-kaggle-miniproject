import numpy as np
from numpy import genfromtxt, linalg
import sklearn
from sklearn import preprocessing, ensemble, feature_selection
from matplotlib import pyplot as plt
from sklearn.externals import joblib

train_file = 'data/training_data_cleaned_for_2012_sub.csv'
test_file = 'data/test_cleaned_for_2012_sub.csv'
outfile = 'data/prediction_2012_sub.csv'



data = genfromtxt(train_file, delimiter=',')

# Trim the first row which contains header information
# and the first three columns which contain irrelevant data
data = data[1:, :]
x_train = data[:, 0:-1]
y_train = data[:, -1]

test_data = genfromtxt(test_file, delimiter=',')
x_test = test_data[1:, :]
print(x_train.shape)
print(x_test.shape)
print("Data successfully loaded from %s and %s" % (train_file, test_file))



# Scale data to have 0 mean and unit variance
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# clf = ensemble.ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5)
# clf = clf.fit(x_train, y_train)
# model = feature_selection.SelectFromModel(clf, prefit=True, threshold='mean')
# x_train = model.transform(x_train)
# x_test = model.transform(x_test)


# Train Gradient Boosting Classifier
print("Begin training Gradient Boosting...")
rf = ensemble.GradientBoostingClassifier(n_estimators=500,
                                         min_samples_leaf=15,
                                         learning_rate=0.05,)
rf.fit(x_train, y_train)

trng_acc = rf.score(x_train, y_train)

#print("leaf size: %d" %leaf_size)
#print("estimators: %d" %num_e)
print("Training accuracy: %f" %trng_acc)

prediction = rf.predict(x_test)

np.savetxt(outfile, prediction)
print("Done! Predictions saved to %s" % outfile)
