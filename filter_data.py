# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:57:44 2022

@author: leoes
"""

import pandas as pd
from collections import OrderedDict
import numpy as np

# Read Dataframe 
calc_training_data_file = pd.read_csv('calc_case_description_train_set.csv',  index_col=False)

#patien id filter
patient_id = calc_training_data_file["patient_id"].tolist() # Get list of patients
patient_id = set(patient_id)    # Delete duplicates in patients
patient_list = list(patient_id)   # Convert to list again

# Read again Dataframe because I had problems with patients
calc_training_data_file = pd.read_csv('calc_case_description_train_set.csv',  index_col=0)

patients_filtered = []  

# Cicle to evaluate if there are CC and MLO in every case
for patient in patient_list:
    x = calc_training_data_file.loc[[patient]]
    CC  = True if 'CC'  in x.values else False
    MLO = True if 'MLO' in x.values else False
    if CC == True and MLO == True:
        patients_filtered.append(patient)

patients_filtered = sorted(patients_filtered) # Sort the list

# Datframe Filtered with the correct cases
df = calc_training_data_file.loc[patients_filtered]







# Filtering individually cases
# x = calc_training_data_file.loc[["P_00013"]]
# True if 'CC' in x.values else False
# True if 'MLO' in x.values else False

# True if calc_training_data_file.isin(['P_00005', 'RIGHT', 'LEFT']) else False