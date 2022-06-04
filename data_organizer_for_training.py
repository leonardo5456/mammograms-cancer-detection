# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 22:42:59 2022

@author: leoes
"""

import pandas as pd

df = pd.read_csv('Caracteristicas_CC.csv')
#calc_training_data_file = pd.read_csv('calc_case_description_train_set.csv')


# patient id filter
# patient_id = calc_training_data_file['patient_id'].tolist()
# patient_id = set(patient_id)    # Delete duplicates in patients
# patient_list = list(patient_id) # Convert to list again
# patient_list = sorted(patient_list)


# Read again Dataframe because I had problems with patients
calc_training_data_file = pd.read_csv('calc_case_description_train_set.csv',  index_col=0)

# Deleting columns
df_filtered = df.dropna(subset=['ROI'])     # Detele NaN values in df
df_filtered = df_filtered.drop(['ROI', 'n ROIs'], axis=1)   # Delete columns 'ROI' and 'n ROIs'

# patient id filter
patient_id = df_filtered['ID'].tolist()
patient_id = [i[:7] for i in patient_id] # taking just ID patient
patient_id = set(patient_id)    # Delete duplicates in patients
patient_list = list(patient_id) # Convert to list again
patient_list = sorted(patient_list)

# for patient in patient_list:
#     x = calc_training_data_file.loc[[patient]]
    
x = calc_training_data_file.loc[patient_list] # find all patients in the list
y = x['assessment'].tolist()

new_df = x.assign(assessment = y) # add a new column to the dataframe



'''
funciona pero en este caso no, porque toma en cuenta todos los valores ya con 
todo procesado y en este caso solo tenemos fracciones de csv.'''
