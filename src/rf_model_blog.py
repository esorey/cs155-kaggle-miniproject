from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV


data_file = 'data/training_data_cleaned1.csv'
data = genfromtxt(data_file, delimiter=',')
print("Data loaded from " + data_file + '...')

# Trim the first row which contains header information
# and the first three columns which contain irrelevant data
data = data[1:, :]
X = data[:, :-1]
Y = data[:, -1]
print(Y[:50])
# partition = 55000
# x_test = x_train[partition:, :]
# y_test = y_train[partition:]
# x_train = x_train[0:partition, :]
# y_train = y_train[0:partition]

forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,],
                 'n_estimators': [500,],
                 'criterion': ['gini', 'entropy']
                 }

cross_validation = KFold(n_splits=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation,
                           verbose=2)

print("Beginning grid search over parameters...")
grid_search.fit(X, Y)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
