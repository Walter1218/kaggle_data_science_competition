"""
Base Plus Model
The Base Model only use numeric data for prediction tasks,
here, we will use all features to improve the results.
Author: Walter Yoda
Date:12/06/16
# rmse score 0.13463
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
        #print(label,score)
        if(score >= thread):
            print(label,score)
            selected_feature.append(label)
    return selected_feature

def feature_regression_within_drop(data, features, thread = 0.7):
    #data = data.drop(features, axis = 1)
    selected_features = []
    for i in range(len(features)):
        label = features[i]
        explorer_log_label = data[label]
        explorer_log_data = data.drop(features, axis =1)
        X_train, X_test, y_train, y_test = train_test_split(explorer_log_data, explorer_log_label, test_size = 0.25, random_state = 42)
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 42)
        regressor.fit(X_train, y_train)
        score = regressor.score(X_test, y_test)
        #print(label,score)
        if(score >= thread):
            print(label,score)
            selected_features.append(label)
    return selected_features

def TukeyOutlier(data,label):
    outlier = []
    log_data = data
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
        #display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
        outlier.append(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index)
    out = []
    for i in range(len(outlier)):
        for j in range(len(outlier[i])):
            if outlier[i][j] not in out:
                out.append(outlier[i][j])
    print(len(out))
    outliers  = out
    # remove outlier
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
    good_label = label.drop(log_data.index[outliers]).reset_index(drop = True)
    return good_data,good_label

def regression_target(data,label,test):
    dtrain = xgb.DMatrix(data, label)
    dtest = xgb.DMatrix(test)
    params = {"max_depth":6, "eta":0.1}
    model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1) #the params were tuned using xgb.cv
    model_xgb.fit(data, label)
    xgb_preds = np.expm1(model_xgb.predict(test))
    return xgb_preds

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
selected_features2 = feature_regression_within_drop(log_train, selected_features)
new_log_train = log_train.drop(selected_features2, axis = 1)
print(new_log_train.shape)
#good_data,good_label = TukeyOutlier(new_log_train,log_label)
#print(good_data.shape,good_label.shape)
new_log_test = log_test.drop(selected_features2, axis = 1)
preds = regression_target(new_log_train, log_label, new_log_test)
solution = pd.DataFrame({"Id":test.Id, "SalePrice":preds})
solution.to_csv("fourth_sub.csv", index = False)
