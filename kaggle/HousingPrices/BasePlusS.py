"""
Base Plus S Model
The Base Plus 3 Model only got best rmse score is 0.12435(0.5Lasso,0.5XGBoost),
here, we will try to improve the results.
Author: Walter Yoda
Date:12/08/16
If I'm not using regression within drop for 80 features, only do it with regression methods to
fill None data, may can improve this model more.

fill all R^2 SCORE > 0.7, RMSE: 0.12371
No fill,RMSE: 0.12309
It means we need not doing regression within features drop here.
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
import pylab
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV, LassoLarsCV, Ridge
from sklearn import preprocessing
import xgboost as xgb
import itertools
from operator import itemgetter

def read_file(data_file_address):
    return pd.read_csv(data_file_address)

# Outlier data remove, handle for missing data. the old methods used for missing data is just filling with median & frequent data
# here we need change this method.
def feature_encoder_split(data,test_length = 0):
    print(test_length[0])
    # data processing, filling & washing
    total_missing=data.isnull().sum()
    num_of_data = data.shape[0]
    to_delete=total_missing[total_missing>(num_of_data/3.)]
    print(to_delete)
    print(len(to_delete))
    print(to_delete.index.tolist())
    delete_list = to_delete.index.tolist()
    print(data.shape)
    data.drop(delete_list,axis=1, inplace=True)
    #print(data.shape)
    #data = pd.DataFrame(n_data)
    print("data shape is",data.shape)
    # got numeric features & categorical features
    numerical_features=data.select_dtypes(include=["float","int","bool"]).columns.values
    categorical_features=data.select_dtypes(include=["object"]).columns.values
    print(categorical_features,len(categorical_features))
    print(numerical_features,len(numerical_features))
    #median_value = np.zeros((1,data.shape[1]))
    for feature in numerical_features:
        data[feature].fillna(data[feature].mean(), inplace = True)
    for feature in categorical_features:
        data[feature].fillna(data[feature].value_counts().idxmax(), inplace = True)
    total_missing=data.isnull().sum()
    to_delete=total_missing[total_missing>(num_of_data/3.)]
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

# Not good way to find the outlier, or just remove frequent outlier data?
def TukeyOutlier(data,label,num_features):
    outlier = []
    #num_data = data.select_dtypes(include=["float","int","bool"]).columns.values
    log_data = data[num_features]
    print("data shape is",log_data.shape)
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
    # TODO
    # change this outliers remove method to others, e.g. just remove by the frequent
    for i in range(len(outlier)):
        for j in range(len(outlier[i])):
            if outlier[i][j] not in out:
                out.append(outlier[i][j])
    print(len(out))
    outliers  = out
    # remove outlier
    good_data = data.drop(log_data.index[outliers]).reset_index(drop = True)
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

# Parameters Fitting for the best Model Performs
def Parameters_Fitting(target,train,label,test,features_name,n_folds,etas,max_depths,boost_round,stopping_rounds,subsample = 1,test_size = 0.2):
    scores = np.ndarray((len(etas)*len(max_depths),4))
    colsample_bytree = 1
    for eta,max_depth in list(itertools.product(etas, max_depths)):
        params = {
            "objective": "reg:linear",
            "booster" : "gbtree",
            "eval_metric": "rmse",
            "eta": eta, # shrinking parameters to prevent overfitting
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample, # collect 80% of the data only to prevent overfitting
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": 42,
        }
        #target = list(label.columns.values)
        frame = [train, label]
        data = pd.concat(frame, axis = 1)
        #print(data.shape)
        kf = KFold(len(data), n_folds=n_folds)
        test_prediction=np.ndarray((n_folds,len(test)))
        fold=0
        fold_score=[]
        i = 0
        for train_index, cv_index in kf:
            X_train, X_valid    = data[features_name].as_matrix()[train_index], train[features_name].as_matrix()[cv_index]
            y_train, y_valid    = data[target].as_matrix()[train_index], data[target].as_matrix()[cv_index]
            dtrain = xgb.DMatrix(X_train, y_train) # DMatrix are matrix for xgboost
            dvalid = xgb.DMatrix(X_valid, y_valid)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # list of things to evaluate and print
            gbm = xgb.train(params, dtrain, boost_round, evals=watchlist, early_stopping_rounds=stopping_rounds, verbose_eval=True) # find the best score
            print("Validating...")
            check = gbm.predict(xgb.DMatrix(X_valid)) # get the best score
            score = gbm.best_score
            print('Check last score value: {:.6f}'.format(score))
            fold_score.append(score)
            importance = gbm.get_fscore()
            importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
            print('Importance array for fold {} :\n {}'.format(fold, importance))
            np.save("features_importance",importance)
            print("Predict test set...")
            prediction=gbm.predict(xgb.DMatrix(test[features_name].as_matrix()))
            test_prediction[fold]=prediction
            fold = fold + 1
        mean_score=np.mean(fold_score)
        print("Mean Score : {}, eta : {}, depth : {}\n".format(mean_score,eta,max_depth))
        scores[i][0]=eta
        scores[i][1]=max_depth
        scores[i][2]=mean_score
        scores[i][3]=np.std(fold_score)
        i+=1
    final_prediction=test_prediction.mean(axis=0)
    df_score=pd.DataFrame(scores,columns=['eta','max_depth','mean_score','std_score'])
    print ("df_score : \n", df_score)
    return final_prediction, mean_score

# get the name of each features in the data
def get_features(data,manual_remove_list = 'Id'):
    dataval = list(data.columns.values) # list train features
    dataval.remove(manual_remove_list) # remove non-usefull id column
    return dataval

def get_num_features(data):
    return data.select_dtypes(include=["float","int","bool"]).columns.values

train_file = "train.csv"
test_file = "test.csv"
train = read_file(train_file)
test = read_file(test_file)
print(train.shape,test.shape)
label = train['SalePrice']
new_train = train.drop(['SalePrice'], axis = 1 )
num_features = get_num_features(new_train)
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
# before do the regression analysis, we need do outlier data remove first
# I try to remove the outlier data, but the rmse results is 0.16000, so
# this step may need remove
#log_train, log_label = TukeyOutlier(log_train, log_label, num_features)
print(log_train.shape,log_label.shape)
# feature regression for all input training data
selected_features = feature_regression(log_train)
print(selected_features)
selected_features2 = feature_regression_within_drop(log_train, selected_features)
print(selected_features2)
#new_log_train = log_train.drop(selected_features2, [selected_features2]
new_log_train = log_train
print(new_log_train.shape)
#good_data,good_label = TukeyOutlier(new_log_train,log_label)
#print(good_data.shape,good_label.shape)
#new_log_test = log_test.drop(selected_features2, axis = 1)
new_log_test = log_test
features_name = get_features(new_log_train)
#features_name = selected_features2
print(features_name)
#"""
# list of parameters
etas = [0.01]
max_depths = [3]#3,6
colsample_bytree = 1
#increase this one for small eta
boost_round = 6000
stopping_rounds = 500
n_folds=12
target = 'SalePrice'
print("start to find the best parameters for this model")
#final_prediction_xgboost, mean_score_xgboost = Parameters_Fitting(target, new_log_train,log_label,new_log_test,features_name,n_folds,etas,max_depths,boost_round,stopping_rounds,subsample = 1,test_size = 0.2)

# update Lasso methods,0.0003,0.00009
lasso = Lasso(alpha = 0.0000009).fit(new_log_train, log_label)
final_prediction_lasso = lasso.predict(new_log_test)

ridge = Ridge(alpha=0.06, solver='auto')
ridge.fit(new_log_train, log_label)
final_prediction_ridge = ridge.predict(new_log_test)

final_prediction = np.exp(final_prediction_xgboost) + np.exp(final_prediction_lasso) + np.exp(final_prediction_ridge)
#fin = np.exp(final_prediction_lasso)
solution = pd.DataFrame({"Id":test.Id, "SalePrice":(final_prediction/3.)})
solution.to_csv("t23_sub.csv", index = False)
#"""
