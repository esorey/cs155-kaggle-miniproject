
# coding: utf-8

# In[311]:

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[312]:

orig_train_data = pd.read_csv("data/train_2008.csv")
train_data = orig_data.copy()

orig_test_data = pd.read_csv("data/test_2008.csv")
test_data = orig_test_data.copy()


# In[313]:

train_data_orig_cols = list(orig_train_data.columns) # original columns, with target PES1 on the end
target_var = orig_train_data['PES1']
target_var_bin = target_var.apply(lambda x: 0 if x == 2 else x)
train_data.drop('PES1', axis=1, inplace=True)


# In[314]:

# Drop allocation flag features
allocation_flag_feats = [col for col in train_data.columns if col[:2] == 'PX']
train_data.drop(allocation_flag_feats, axis=1, inplace=True)
test_data.drop(allocation_flag_feats, axis=1, inplace=True)

# Drop "weight" features
weight_feats = [col for col in train_data.columns if col[-3:] == 'WGT']
train_data.drop(weight_feats, axis=1, inplace=True)
test_data.drop(weight_feats, axis=1, inplace=True)

# Drop some other columns that are "bad"
bad_feats = ['QSTNUM', 'PRERNWA', 'PRERNHLY', 'PEERNH1O', 'HRHHID2', 'GTCSA', 'GTCO', 'HUINTTYP', 'HURESPLI', 'HRMIS',
            'PRDTOCC1', 'PRFAMREL', 'PUSLFPRX', 'OCCURNUM', 'PULINENO', 'PRMJOCC1', 'PRCHLD', 'GTCBSA', 'HRLONGLK',]
train_data.drop(bad_feats, axis=1, inplace=True)
test_data.drop(bad_feats, axis=1, inplace=True)


# In[315]:

### First pass - use mean for NaN. Later we'll do intelligent filling on important features

# Replace negative values (all forms of N/A) with NaN
for feat in train_data.columns:
    train_data[feat] = train_data[feat].apply(lambda x: np.NaN if x < 0 else x)
    test_data[feat] = test_data[feat].apply(lambda x: np.NaN if x < 0 else x)
    
# Replace NaN with the mean of the column
for feat in train_data.columns:
    train_data[feat].fillna(train_data[feat].mean(), inplace=True)
    test_data[feat].fillna(test_data[feat].mean(), inplace=True)
    
# Check for columns that are all NaN, and delete them
all_nan_cols = []
for feat in train_data.columns:
    if np.all(np.isnan(train_data[feat])):
        all_nan_cols.append(feat)
train_data.drop(all_nan_cols, axis=1, inplace=True)
test_data.drop(all_nan_cols, axis=1, inplace=True)


# In[316]:

# Dummify categorical vars
to_dummy = ['GEREG', 'HUBUS', 'PTDTRACE', 'PENATVTY', 'PUABSOT', 'PEIO1COW', 'HUFINAL', 'GESTCEN', 'GESTFIPS',
            'PEIO1ICD', 'PEIO1OCD', 'PEIO2ICD', 'PEIO2OCD', 'PRCITSHP', 'PUDIS', 
           'PRABSREA', 'PRWKSTAT', 'HUPRSCNT', 'PERRP', 'GTCBSAST', 'PRMJOCGR', 'HRHTYPE', ]

train_dummy_df = pd.DataFrame()
test_dummy_df = pd.DataFrame()

for var in to_dummy:
    train_dummy_vars = pd.get_dummies(train_data[var], prefix=var)
    train_dummy_df = pd.concat([train_dummy_df, train_dummy_vars], axis=1)
    
    test_dummy_vars = pd.get_dummies(test_data[var], prefix=var)
    test_dummy_df = pd.concat([test_dummy_df, test_dummy_vars], axis=1)
    
# Drop the original categorical variables
train_data.drop(to_dummy, axis=1, inplace=True)
test_data.drop(to_dummy, axis=1, inplace=True)


# In[317]:

# Add dummy vars to the data
train_data = pd.concat([train_data, train_dummy_df], axis=1)
test_data = pd.concat([test_data, test_dummy_df], axis=1)


# In[318]:

# Fit a classifier for evaluating most important features
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train_data.ix[:, 3:], target_var)


# In[319]:

# Make a dataframe of the features and their importances
features = pd.DataFrame()
features['feature'] = train_data.columns[3:]
features['importance'] = clf.feature_importances_


# In[320]:

# Take a look at the most important features
features.sort(['importance'],ascending=False).head(200)


# In[325]:

# Pull out the most important features
model = SelectFromModel(clf, prefit=True)
train_data_selected_feats = model.transform(train_data.ix[:, 3:])


# In[328]:

# Kinda hacky, but we need some way of getting just the selected features in the test set (selecting features kills
# the column names).
imp_mean = features.importance.mean()
keep_feats = []
for f in features['feature']:
    if float(features[features['feature'] == f]['importance']) >= imp_mean:
        keep_feats.append(f)


# In[332]:

# Keep hacking....
keep_feats = [f for f in keep_feats if f in test_data.columns]


# In[337]:

# Finally get the features we need
train_data_selected_feats = train_data[keep_feats]
test_data_selected_feats = test_data[keep_feats]


# In[340]:

# Write out the data sets
train_data_selected_feats.to_csv("data/training_data_cleaned2.csv", index=False)
test_data_selected_feats.to_csv("data/test_data_cleaned2.csv", index=False)


# In[ ]:




# In[ ]:




# In[341]:

######### Havent used this stuff yet

# For numerics, define a find_min_entropy_split() function, and use that on
# negative valued responses. Negative valued responses indicate that the
# survey participant did not respond in some capacity.
def get_split_points(col):
    '''Get the points between the unique values in col. col is a pandas series'''
    unique = col.unique()
    sort = sorted(unique)
    splits = []
    for i in range(len(sort) - 1):
        splits.append(0.5 * (sort[i] + sort[i + 1]))
    return splits

def get_entropy(arr):
    '''Get the entropy of an array. Assumes the values of the array are 0/1'''
    frac_pos = sum(arr) / len(arr)
    size = len(arr)
    if frac_pos == 0 or frac_pos == 1: # Workaround for defining 0 * log(0) = 0
        return 0.0
    return -size * (frac_pos * np.log(frac_pos) + (1 - frac_pos) * np.log(1 - frac_pos))

def find_min_entropy_split(col):
    '''Find the threshold in a column of data that a decision tree would choose using
    the entropy impurity measure. Return that threshold, and flag indicating which side of
    the split has higher entropy. flag=1 => the datapoints above the split have higher entropy, and
    flag=-1 => the datapoints below the split have higher entropy.'''
    split_points = get_split_points(col)
    min_entropy = np.inf
    min_entropy_split = None
    flag = 0
    for split in split_points:
        bools = col > split
        above = [i for i in range(len(col)) if bools[i]]
        below = [i for i in range(len(col)) if not bools[i]]
        voter_status_above = [target_var_bin[i] for i in above]
        voter_status_below = [target_var_bin[i] for i in below]
        entropy_above = get_entropy(voter_status_above)
        entropy_below = get_entropy(voter_status_below)
        entropy_total = entropy_above + entropy_below
        if entropy_total < min_entropy:
            min_entropy = entropy_total
            min_entropy_split = split
            if entropy_above > entropy_below:
                flag = 1
            else:
                flag = -1
    return min_entropy_split, flag

def get_fill_value(col):
    '''First find the split that minimizes entropy (ie the split chosen by a tree), then return the mean
    of the half of the split that has higher entropy.'''
    split, flag = find_min_entropy_split(col)
    assert flag != 0 # If the flag is zero, something went wrong
    if flag == 1:
        higher_entropy_side = col[col > split]
    else:
        higher_entropy_side = col[col <= split]
    return np.mean(higher_entropy_side)


# In[342]:

numeric_cols = [col for col in train_data.columns if ('_' not in col and col not in to_dummy)]
numeric_cols = numeric_cols[3:]


# In[343]:

# For each feature, apply find_max_entropy_split and fill the negative 
# values with the mean of the higher entropy split


# In[ ]:



