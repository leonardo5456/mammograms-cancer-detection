# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 23:41:34 2022

@author: leoes


@Description:
    
    
@Data:
    BI-RADS
    cases for training:
        0 = 63 cases
        2 = 476 cases
        3 = 86
        4 = 678
        5 = 145
"""

import pandas as pd

def reducing_cases(datasheet, birad, n_data):
    c = datasheet[datasheet['assessment'] == birad]
    c = c.sample(85, random_state = n_data)
    return c
    

#%%
''' Preparin data'''
# Read Dataseet
datasheet_train = pd.read_csv('data/data_for_training_features.csv',engine='python', index_col=0)
datasheet_test = pd.read_csv('data/data_for_TEST_features.csv',engine='python', index_col=0)

#c_0 = datasheet_train[datasheet_train['assessment'] == 0]
# c_2 = datasheet_train[datasheet_train['assessment'] == 2]
# c_3 = datasheet_train[datasheet_train['assessment'] == 3]
c_4 = datasheet_train[datasheet_train['assessment'] == 4]
c_4_1 = c_4.sample(85, random_state = 678)
# c_5 = datasheet_train[datasheet_train['assessment'] == 5]

c_0 = datasheet_train[datasheet_train['assessment'] == 0]
c_2 = reducing_cases(datasheet_train, 2, 476)
c_3 = reducing_cases(datasheet_train, 3, 86)
c_4 = reducing_cases(datasheet_train, 4, 678)
c_5 = reducing_cases(datasheet_train, 5, 145)

data = pd.concat([c_0,c_2,c_3,c_4,c_5])
