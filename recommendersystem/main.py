from __future__ import division
import math
import numpy as np
import random
import UserCF
import utility

train_data = 'mldataset/ratings.csv'
data = utility.LoadData_S1(train_data)
data.to_csv("data.csv", index = False, header = False )
#train, test = utility.SplitData(data, M = 8, k = 80, seed = 25)
#print(len(train))
#TrainData = utility.LoadData_S2(train)
#TestData = utility.LoadData_S2(test)
#Train = utility.TransformData(TrainData)
#print(Train)
#W = UserSimilarity_V0(Train)
#print(W)
#print(test)
