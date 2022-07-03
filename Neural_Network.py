# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:37:34 2022

@author: leoes

@Description:
        Neural network application for breast cancer detection.
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

''' Preparin data'''
# Read Dataseet
datasheet_train = pd.read_csv('data/data_for_training_features.csv',engine='python', index_col=0)
datasheet_test = pd.read_csv('data/data_for_TEST_features.csv',engine='python', index_col=0)

# Predictor variable
x_train = datasheet_train.iloc[:,0:6]   # takes all columns except "assessment"
x_test = datasheet_test.iloc[:,0:6]     # takes all columns except "assessment"

# Variable to predict
y_train = datasheet_train.iloc[:,6] # takes just "enfermedad" to predict 
y_test = datasheet_test.iloc[:,6]   # takes just "enfermedad" to predict
y_train = y_train.astype(str)
y_test = y_test.astype(str) 

