import math
import numpy as np
import random
import os
import pandas as pd

"""
utility function for recommender system
"""

def RMSE(records):
    return math.sqrt(sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records]) / float(len(records)))

def MAE(records):
    return sum([abs(rui - pui) for u, i, rui, pui in records]) / float(len(records))

def LoadData_S1(filename):
    prefs = []
    count = 0
    try:
        with open(filename) as train:
            for line in train:
                #ignore the first line
                if(count == 0):
                    count += 1
                    continue
                userId, movieId, rating, time = line.split(',')
                prefs.append([userId, movieId, rating])
    except IOError as err:
        print('File error: ' + str(err))
    df = pd.DataFrame(prefs)
    return df

def LoadData_S2(data):
    prefs = {}
    for user, item, rating in data:
        prefs.setdefault(user,{})
        prefs[user][item] = float(rating)
    return prefs

def TransformData(data):
    results = {}
    for person in data:
        for item in data[person]:
            results.setdefault(item, {})
            results[item][person] = data[person][item]
    return results

def SplitData(data, M = 8, k = 80, seed = 25):
    test = {}
    train = {}
    random.seed(seed)
    for user, item, rating in data:
        if random.randint(0, M) == k:
            test.setdefault(user, {})
            test[user][item] = rating
            #test.append([user, item])
        else:
            train.setdefault(user, {})
            train[user][item] = rating
            #train.append([user, item])
    return train, test
