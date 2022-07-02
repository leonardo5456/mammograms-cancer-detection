# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:54:01 2022

@author: leoes


@ Description:
    This program is to accommodate the data in a new dataframe to execute it in 
    a program with desition trees
"""

import pandas as pd

'''Data for training'''
# read data processed with features 
# data_1 = pd.read_csv('data/00_Caracteristicas_R_CC.csv')
# data_2 = pd.read_csv('data/01_Caracteristicas_L_CC.csv')
# data_3 = pd.read_csv('data/02_Caracteristicas_R_MLO.csv')
# data_4 = pd.read_csv('data/03_Caracteristicas_L_MLO.csv')
# # Read original data 
# calc_training_data_file = pd.read_csv('data/data_filtered.csv')

'''Data for testing'''
# read data processed with features 
data_1 = pd.read_csv('data/00_TEST_Caracteristicas_R_MLO.csv')
data_2 = pd.read_csv('data/01_TEST_Caracteristicas_L_MLO.csv')
data_3 = pd.read_csv('data/02_TEST_Caracteristicas_R_CC.csv')
data_4 = pd.read_csv('data/03_TEST_Caracteristicas_L_CC.csv')
# Read original data 
calc_training_data_file = pd.read_csv('data/calc_case_description_test_set.csv')


# patient id filter
patient_id_2 = calc_training_data_file['patient_id'].tolist()
patient_id_2 = [i[:7] for i in patient_id_2] # taking just ID patient
# training file
#calc_training_data_file = pd.read_csv('data/data_filtered.csv',  index_col=0)
# test file
calc_training_data_file = pd.read_csv('data/calc_case_description_test_set.csv',  index_col=0)



# Concat all Dataframes
data = pd.concat([data_1,data_2,data_3,data_4])
# sort data by ID
data = data.sort_values('ID')

# Deleting columns
df_filtered = data.dropna(subset=['ROI'])     # Detele NaN values in df
df_filtered = df_filtered.drop(['ROI', 'n ROIs'], axis=1)   # Delete columns 'ROI' and 'n ROIs'

# ''' it found a problem with the patient P_00474, it is neccesary t delete it for training cases'''
# # P_00474 has a missing data
# P_00474 = ['P_00474']
# # Deleting P_00474, wich present problems from data_filter
# df_filtered = df_filtered[df_filtered["ID"] != "P_00474_LEFT_CC_1"]
# # Deleting P_00474 in Calc_training_data_file
# calc_training_data_file = calc_training_data_file.drop(['P_00474'])

''' it found a problem with the patient P_00353, it is neccesary t delete it for training cases'''
P_00353 = ['P_00353']
# Deleting P_00535, wich present problems from data_filter
df_filtered = df_filtered[(df_filtered["ID"] != "P_00353_LEFT_CC_1") & (df_filtered["ID"] != "P_00353_LEFT_MLO_1") ]
# Deleting P_00474 in Calc_training_data_file
calc_training_data_file = calc_training_data_file.drop(['P_00353'])


''' you need to run this part first if you see data inconsistency with the main data 
with the filtered one'''
# patient id filter
patient_id_1 = df_filtered['ID'].tolist()
patient_id_1 = [i[:7] for i in patient_id_1] # taking just ID patient
patient_id = set(patient_id_1)    # Delete duplicates in patients
patient_list = list(patient_id) # Convert to list again
patient_list = sorted(patient_list)


patients_counted_1 = []
patients_counted_2 = []

for patient in patient_list:    
    patien_id_counted = patient_id_1.count(patient)
    patients_counted_1.append(patien_id_counted)

for patient in patient_list:    
    patien_id_counted = patient_id_2.count(patient)
    patients_counted_2.append(patien_id_counted)
    
    
# Convert list in diccionary with repetitions
dic_patient_id_1 = dict(zip(patient_list, patients_counted_1))
dic_patient_id_2 = dict(zip(patient_list, patients_counted_2))


# Compare the dictionaries to find de missing data
for i,j  in zip(dic_patient_id_1,dic_patient_id_2 ):
    #print(f'{dic_patient_id_1[i]}, {dic_patient_id_2[j]}')
    #if (i != j):
    if dic_patient_id_1[i] != dic_patient_id_2[j]:
        print (f'{i} is not equal to {j}')


x = calc_training_data_file.loc[patient_list] # find all patients in the list
y = x['assessment'].tolist()

new_df = df_filtered.assign(assessment = y) # add a new column to the dataframe
# Guarda datos en CSV:
#new_df.to_csv('data/data_for_training_features.csv', header=True, index=False)
