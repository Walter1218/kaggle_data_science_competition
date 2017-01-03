
# coding: utf-8

# In[1]:

"""
My Base Model of HousingPrices Prediction Task
Author: Walter Yoda
Date:12/02/16
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic("config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
get_ipython().magic('matplotlib inline')


# In[2]:

#read data from csv file
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:

# y Labels
label = train['SalePrice']
# X features
features = train.drop(['SalePrice'], axis =1)


# In[4]:

#numeric_features = features._get_numeric_data()
# log transform for label data.
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"log(price + 1)":np.log1p(label), "price":label})
prices.hist()
log_label = np.log1p(label)


# In[5]:

from scipy.stats import skew
import pylab
# The Base model only using numeric data
# before we build the Base Model , we should also doing log transform for the features data
numeric_feats = features.dtypes[features.dtypes != "object"].index
skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
log_features = features
log_features[skewed_feats] = np.log1p(features[skewed_feats])
numeric_log_features = log_features._get_numeric_data()
print(numeric_log_features.shape)


# In[6]:

# fill None data by using mean data
numeric_log_features = numeric_log_features.fillna(numeric_log_features.mean())


# In[38]:

pylab.clf()
axs = pd.scatter_matrix(numeric_log_features, alpha = 0.3, figsize = (140,80), diagonal = 'kde');
pylab.savefig("res.png")


# In[7]:

# dataframe types
features.dtypes


# In[8]:

numeric_log_features.dtypes


# In[9]:

# regression explorer, 36 features within one single label
explorer_data = ['Id','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']
explorer_log_label = log_label
from sklearn.cross_validation import train_test_split
# split data
X_train, X_test, y_train, y_test = train_test_split(numeric_log_features, explorer_log_label, test_size = 0.25, random_state = 42)
from sklearn.tree import DecisionTreeRegressor
# TODO：create regression
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train, y_train)
# TODO：output score
score = regressor.score(X_test, y_test)
print(score)


# In[10]:

# other regression explorer, drop one single feature as label to see the output score
for i in range(len(explorer_data)):
    label = explorer_data[i]
    explorer_log_label = numeric_log_features[label]
    explorer_log_data = numeric_log_features.drop([label], axis =1)
    X_train, X_test, y_train, y_test = train_test_split(explorer_log_data, explorer_log_label, test_size = 0.25, random_state = 42)
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 42)
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print(label,score)


# label results
# Id -1.15651044538
# MSSubClass 0.330963908219
# LotFrontage 0.0607400339408
# LotArea 0.356923986951
# OverallQual 0.539688626994
# OverallCond 0.0381132601175
# YearBuilt 0.805894976021
# YearRemodAdd 0.450245903345
# MasVnrArea -0.0872715047805
# BsmtFinSF1 0.825717048199
# BsmtFinSF2 0.19417350471
# BsmtUnfSF 0.552390843684
# TotalBsmtSF 0.970761469481
# 1stFlrSF 0.913925010969
# 2ndFlrSF 0.880153969916
# LowQualFinSF -0.503201023403
# GrLivArea 0.950294104228
# BsmtFullBath 0.0687242798354
# BsmtHalfBath -0.597554426914
# FullBath 0.47032113098
# HalfBath 0.513954391611
# BedroomAbvGr 0.417007911155
# KitchenAbvGr 0.217183716406
# TotRmsAbvGrd 0.598252674303
# Fireplaces -0.256669550797
# GarageYrBlt 0.639567860691
# GarageArea 0.7454709875
# WoodDeckSF -0.679650392379
# OpenPorchSF -0.221916248444
# EnclosedPorch -0.196475411333
# 3SsnPorch -1.83327944052
# ScreenPorch -0.96709521633
# PoolArea -2.00893523235
# MiscVal -2.69622643444
# MoSold -0.934885617552
# YrSold -0.818390465763

# 'YearBuilt', 'BsmtFinSF1', '1stFlrSF' , '2ndFlrSF', 'GrLivArea' is bigger than 0.80

# In[11]:

#'1stFlrSF' this feature's r62 score is not good when we drop all 5 features['YearBuilt', 'BsmtFinSF1', '1stFlrSF' , '2ndFlrSF', 'GrLivArea']
target_feature = ['YearBuilt', 'BsmtFinSF1', '2ndFlrSF', 'GrLivArea']
for i in range(len(target_feature)):
    label = target_feature[i]
    explorer_log_label = numeric_log_features[label]
    explorer_log_data = numeric_log_features.drop(target_feature, axis =1)
    X_train, X_test, y_train, y_test = train_test_split(explorer_log_data, explorer_log_label, test_size = 0.25, random_state = 42)
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 42)
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print(label,score)
"""
this 4 features can used other features to represent, so it means we can drop this 4 numeric data here.
"""


# In[12]:

# regression explorer view 1
explorer_log_label = log_label
new_numeric_data = numeric_log_features.drop(['YearBuilt', 'BsmtFinSF1', '2ndFlrSF', 'GrLivArea'], axis = 1)
from sklearn.cross_validation import train_test_split
# split data
X_train, X_test, y_train, y_test = train_test_split(new_numeric_data, explorer_log_label, test_size = 0.25, random_state = 42)
from sklearn.tree import DecisionTreeRegressor
# TODO：create regression
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train, y_train)
# TODO：output score
score = regressor.score(X_test, y_test)
print(score)


# In[13]:

from IPython.display import display
def TukeyOutlier():   
    outlier = []
    log_data = new_numeric_data
    # for each single feature, find the outlier based on Turkey IQR theory
    for feature in log_data.keys():
    
        # 25th Q1
        Q1 = np.percentile(log_data[feature],25)
    
        # 75th Q3
        Q3 = np.percentile(log_data[feature],75)
    
        # Turkey IQR
        step = (Q3 - Q1) * 1.5
    
        # show outlier
        print ("Data points considered outliers for the feature '{}':".format(feature))
        display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
        outlier.append(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index)
    out = []
    for i in range(len(outlier)):
        for j in range(len(outlier[i])):
            if outlier[i][j] not in out:
                out.append(outlier[i][j])
    print(out)

    outliers  = out

    # remove outlier
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
    good_label = explorer_log_label.drop(log_data.index[outliers]).reset_index(drop = True)
    return good_data,good_label


# In[14]:

good_data, good_label = TukeyOutlier()


# In[15]:

# now , we drop 629 rows data, regression again
from sklearn.cross_validation import train_test_split
# split data
X_train, X_test, y_train, y_test = train_test_split(good_data, good_label, test_size = 0.25, random_state = 42)
from sklearn.tree import DecisionTreeRegressor
# TODO：create regression
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train, y_train)
# TODO：output score
score = regressor.score(X_test, y_test)
print(score)


# In[33]:

# test data
numeric_test = test.dtypes[test.dtypes != "object"].index
skewed_test = test[numeric_test].apply(lambda x: skew(x.dropna()))
skewed_test = skewed_test[skewed_test > 0.75]
skewed_test = skewed_test.index
log_test = test
log_test[skewed_test] = np.log1p(test[skewed_test])
numeric_log_test = log_test._get_numeric_data()
print(numeric_log_test.shape)


# In[61]:

new_test = numeric_log_test.drop(['YearBuilt', 'BsmtFinSF1', '2ndFlrSF', 'GrLivArea'], axis = 1)
# using xgboost
import xgboost as xgb
dtrain = xgb.DMatrix(good_data, label = good_label)
dtest = xgb.DMatrix(new_test)
params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[62]:

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[63]:

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(good_data, good_label)


# In[64]:

xgb_preds = np.expm1(model_xgb.predict(new_test))


# In[65]:

print(xgb_preds.shape)


# In[66]:

solution = pd.DataFrame({"Id":test.Id, "SalePrice":xgb_preds})
solution.to_csv("first_sub.csv", index = False)
# RMSE 0.22

