#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[462]:

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats

# In[463]:


player_df = pd.read_csv('fifa19.csv')


# In[464]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[465]:


player_df = player_df[numcols+catcols]


# In[466]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[467]:


traindf = pd.DataFrame(traindf,columns=features)


# In[468]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[469]:


X.head()


# In[470]:


len(X.columns)


# ### Set some fixed set of features

# In[471]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[472]:


def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    feature_name = X.columns.tolist()
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    # Your code ends here
    return cor_support, cor_feature


# In[473]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[474]:


cor_feature


# ## Filter Feature Selection - Chi-Sqaure

# In[475]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[476]:


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    # Your code ends here
    return chi_support, chi_feature


# In[477]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[478]:


chi_feature


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[479]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[480]:


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature


# In[481]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[482]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[483]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[484]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[485]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[486]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[487]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[488]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# In[489]:


embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')


# In[490]:


embedded_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[491]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[492]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[493]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[494]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[495]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
# Convert boolean values to integers
feature_selection_df['Total'] = feature_selection_df[['Pearson', 'Chi-2', 'RFE', 'Logistics', 'Random Forest', 'LightGBM']].sum(axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[496]:


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    dataset = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    dataset = dataset[numcols+catcols]
    dataset = pd.concat([dataset[numcols], pd.get_dummies(dataset[catcols])],axis=1)
    features = dataset.columns
    dataset = dataset.dropna()
    dataset = pd.DataFrame(dataset,columns=features)
    y = dataset['Overall']>=87
    X = dataset.copy()
    del X['Overall']
    num_feats = 30
    # Your code ends here
    return X, y, num_feats


# In[497]:


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    all_feature_lists = [cor_feature, chi_feature, rfe_feature, embedded_lr_feature, embedded_rf_feature, embedded_lgbm_feature]
    feature_counts = {}
    for features in all_feature_lists:
        for feature in features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    max_votes = max(feature_counts.values())
    best_features = [feature for feature, count in feature_counts.items() if count == max_votes]
    #### Your Code ends here
    return best_features


# In[498]:


best_features = autoFeatureSelector(dataset_path='fifa19.csv', methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# In[499]:


from sklearn.model_selection import train_test_split
X = traindf[best_features]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[500]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))