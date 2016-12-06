"""
Base Plus Model
The Base Model only use numeric data for prediction tasks,
here, we will use all features to improve the results.
Author: Walter Yoda
Date:12/06/16
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
import pylab
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import xgboost as xgb

def read_file(data_file_address):
    return pd.read_csv(data_file_address)

def feature_encoder_split(data,test_length = 0):
    print(test_length[0])
    # got numeric features & categorical features
    numerical_features=data.select_dtypes(include=["float","int","bool"]).columns.values
    categorical_features=data.select_dtypes(include=["object"]).columns.values
    print(categorical_features,len(categorical_features))
    print(numerical_features,len(numerical_features))
    # data processing, filling & washing
    total_missing=data.isnull().sum()
    to_delete=total_missing[total_missing>(1460/3.)]
    print(to_delete)
    print(len(to_delete))
    for feature in numerical_features:
        data[feature].fillna(data[feature].median(), inplace = True)
    for feature in categorical_features:
        data[feature].fillna(data[feature].value_counts().idxmax(), inplace = True)
    total_missing=data.isnull().sum()
    to_delete=total_missing[total_missing>(1460/3.)]
    # Encode categorical features by using sklearn LabelEncoder function
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(data[feature])
        data[feature]=le.transform(data[feature])
    print(len(to_delete))
    # split the train & test data by using test_length attribute
    #print(data.iloc[0])
    data_length = data.shape[0]
    print(data_length)
    #print(data[:1])
    train = data[:(data_length - test_length[0] + 1)]
    test = data[test_length[0]:]
    return train, test

def saving2csv(name,data):
    data.to_csv(name,index = False)

def log_transform(data):
    # doing log transform for input data
    log_data = np.log1p(data)
    return log_data

def feature_regression(data, thread = 0.7):
    numerical_features=data.select_dtypes(include=["float","int","bool"]).columns.values
    print(len(numerical_features))
    selected_feature = []
    for i in range(len(numerical_features)):
        label = numerical_features[i]
        explorer_log_label = data[label]
        explorer_log_data = data.drop([label], axis =1)
        X_train, X_test, y_train, y_test = train_test_split(explorer_log_data, explorer_log_label, test_size = 0.25, random_state = 42)
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 42)
        regressor.fit(X_train, y_train)
        score = regressor.score(X_test, y_test)
        print(label,score)
        if(score >= thread):
            selected_feature.append(label)
    return selected_feature

def regression_target():
    # TODO
    return 0

train_file = "train.csv"
test_file = "test.csv"
train = read_file(train_file)
test = read_file(test_file)
print(train.shape,test.shape)
label = train['SalePrice']
new_train = train.drop(['SalePrice'], axis = 1 )
frame = [new_train, test]
data = pd.concat(frame)
print(data.shape)
n_train, n_test = feature_encoder_split(data,test_length = train.shape)
print(n_train.shape, n_test.shape)
saving2csv("new_train.csv",n_train)
saving2csv("new_test.csv",n_test)
# doing log transform for our train, label, and test data
log_train = log_transform(n_train)
log_label = log_transform(label)
log_test = log_transform(n_test)
# feature regression for all input training data
selected_features = feature_regression(log_train)
print(selected_features)
new_log_train = log_train.drop(selected_features, axis = 1)
print(new_log_train.shape)
