import math
import numpy as np
import random

"""
UserCF.py
"""

def UserSimilarity_V0(train):
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v :
                continue
            W[u][v] = len(train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

def Recommender(user, train, W):
    rank = dict()
    inter_item = train[user]
    for v, wuv in sorted(W[u].items, key = itemgetter(1), reverse = True)[0:k]:
        for i, rvi in train[v].itemd:
            if i in inter_item:
                continue
            rank[i] += wuv * rvi
    return rank
