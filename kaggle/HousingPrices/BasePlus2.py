"""
Base Plus2 Model
The Base Plus Model only got rmse score 0.13463,
here, we will try to improve the results.
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
