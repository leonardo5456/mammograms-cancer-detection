# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:37:34 2022

@author: leoes

@Description:
        Neural network application for breast cancer detection.
@Data:
    BI-RADS
    cases for training:
        0 = 63 cases
        2 = 476 cases
        3 = 86
        4 = 678
        5 = 145
    
    To balance the data, we will take just cases 2,4 and 5
    due there are very few cases in the datasheen and theese cases
    are the ones with the most.
        
"""
#%%
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
#import tensorflowjs as tfjs
#import tensorflow as tf

#%%
# Convert the case in an array [0,2,3,4,5] (complete datasheet)
def str_to_list(option):
    # BI-RADS cases
    # res = [0,2,3,4,5]
    if option == '0':
        res = [1,0,0,0,0]
    elif option == '2':
        res = [0,1,0,0,0]
    elif option == '3':
        res = [0,0,1,0,0]
    elif option == '4':
        res = [0,0,0,1,0]
    elif option == '5':
        res = [0,0,0,0,1]
    else:
        res = ["error"]
        
    return res

# the model makes an prediction, this function select the prediction 
# with a percentaje closest to 1 with cases [0,2,3,4,5]
def detect(predict):
  if   predict[0][0] > predict[0][1] and predict[0][0] > predict[0][2] and predict[0][0] > predict[0][3] and predict[0][0] > predict[0][4]:
    array = [1,0,0,0,0]
    action = "0"
  elif predict[0][1] > predict[0][0] and predict[0][1] > predict[0][2] and predict[0][1] > predict[0][3] and predict[0][1] > predict[0][4]:
    array = [0,1,0,0,0]
    action = "2"
  elif predict[0][2] > predict[0][0] and predict[0][2] > predict[0][1] and predict[0][2] > predict[0][3] and predict[0][2] > predict[0][4]:
    array = [0,0,1,0,0]
    action = "3"
  elif predict[0][3] > predict[0][0] and predict[0][3] > predict[0][1] and predict[0][3] > predict[0][2] and predict[0][3] > predict[0][4]:
    array = [0,0,0,1,0]
    action = "4"
  elif predict[0][4] > predict[0][0] and predict[0][4] > predict[0][1] and predict[0][4] > predict[0][2] and predict[0][4] > predict[0][3]:
    array = [0,0,0,0,1]
    action = "5"
    
  return array, action


# Function for training just with cases [2,4,5]
# Convert the case in an array
def str_to_list_new_cases(option):
    # BI-RADS cases
    # res = [2,4,5]
    if option=='2':
        res = [1,0,0]
    elif option=='4':
        res = [0,1,0]
    elif option=='5':
        res = [0,0,1]
    else:
        res = ["error"]
        
    return res

# the model makes an prediction, this function select the prediction 
# with a percentaje closest to 1 with cases [2,4,5]
def detect_new_cases(predict):
  if predict[0][0] > predict[0][1] and predict[0][0] > predict[0][2]:
    array = [1,0,0]
    action = '2'
  elif predict[0][1] > predict[0][0] and predict[0][1] > predict[0][2]:
    array = [0,1,0]
    action = '4'
  elif predict[0][2] > predict[0][0] and predict[0][2] > predict[0][1]:
    array = [0,0,1]
    action = '5'
  
  return array,action


def reducing_cases(datasheet, birad, n_data, n_reduction):
    c = datasheet[datasheet['assessment'] == birad]
    c = c.sample(n_reduction, random_state = n_data)
    return c
    

#%%
''' Preparin data'''
# Read Dataseet
datasheet_train = pd.read_csv('data/data_for_training_features.csv',engine='python', index_col=0)
datasheet_test = pd.read_csv('data/data_for_TEST_features.csv',engine='python', index_col=0)

# Theese are for the complete cases
# c_4 = datasheet_train[datasheet_train['assessment'] == 4]
# c_4_1 = c_4.sample(85, random_state = 678)
# c_5 = datasheet_train[datasheet_train['assessment'] == 5]

# Cases to testing with all the data
# c_0 = datasheet_train[datasheet_train['assessment'] == 0]
# c_2 = reducing_cases(datasheet_train, 2, 476, 85)
# c_3 = reducing_cases(datasheet_train, 3, 86 , 85)
# c_4 = reducing_cases(datasheet_train, 4, 678, 85)
# c_5 = reducing_cases(datasheet_train, 5, 145, 85)

# Creation of a new datasheet
#datasheet_train = pd.concat([c_0,c_2,c_3,c_4,c_5])


# Cases for testing with cases [2,4,5]
c_2 = reducing_cases(datasheet_train, 2, 476, 200)
c_4 = reducing_cases(datasheet_train, 4, 678, 200)
c_5 = datasheet_train[datasheet_train['assessment'] == 5]

# Creation of a new datasheet for training
datasheet_train = pd.concat([c_2,c_4,c_5])

# For theese cases is necesary filtered the test datasheet with the cases [2,4,5]
t_2 = datasheet_test[datasheet_test['assessment'] == 2]
t_4 = datasheet_test[datasheet_test['assessment'] == 4]
t_5 = datasheet_test[datasheet_test['assessment'] == 5]

# Creation of a new datasheet for testing
datasheet_test = pd.concat([t_2,t_4,t_5])

#%%
# Predictor variable
x_train = datasheet_train.iloc[:,0:6]   # takes all columns except "assessment"
x_test = datasheet_test.iloc[:,0:6]     # takes all columns except "assessment"

# Variable to predict
y_train = datasheet_train.iloc[:,6] # takes just "enfermedad" to predict 
y_test = datasheet_test.iloc[:,6]   # takes just "enfermedad" to predict
y_train = y_train.astype(str)
y_test = y_test.astype(str) 

# it is necessary to conver DataFrame to an array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Convert data in formar [0,0,0,0,0] for cases [0,2,3,4,5]
# target_data_train = list(map(str_to_list, y_train))
# target_data_train = np.array(target_data_train,"float32")
# target_data_test = list(map(str_to_list, y_test))
# target_data_test = np.array(target_data_test,"float32")


# Convert data in formar [0,0,0] for cases [2,4,5]
target_data_train = list(map(str_to_list_new_cases, y_train))
target_data_train = np.array(target_data_train,"float32")
target_data_test = list(map(str_to_list_new_cases, y_test))
target_data_test = np.array(target_data_test,"float32")

#%%
# Create model with keras and tensorflow
model = Sequential()
model.add(Dense(3, input_dim=6, activation='sigmoid'))
model.add(Dense(15, input_dim=7, activation='sigmoid'))
#model.add(Dense(20, input_dim=3, activation='tanh'))
#model.add(Dense(5, activation='selu')) # uncomment when you use all the cases
model.add(Dense(3, activation='selu'))

model.compile(loss='mean_squared_error',
              #optimizer='rmsprop',
              optimizer='adam',
              metrics=['binary_accuracy'])
              #metrics=['accuracy'])
# Fit model
model.fit(x_train, target_data_train, epochs=1500)

#%%
# Evaluates the model 
scores = model(x_train, target_data_train)
print(model.metrics_names[1], scores[1]*100)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(x_train).round())


#%%
#save training
# Convert model to a JSON file
model_json = model.to_json()
with open("data/neural_network/model_c245.json", "w") as json_file:
    json_file.write(model_json)
# Convert the weights to HDF5
model.save_weights("data/neural_network/model_c245.h5")
print("Model saved Correctly!")
 

#%%
# Load json and create the model
json_file = open('data/neural_network/model_c245.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights to the new model
loaded_model.load_weights("data/neural_network/model_c245.h5")
print("Model loaded Correctly.")
 
# Compile the model loaded and it is ready to use
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

#%%
# testin model with individual data

valor = np.array([[38177.0,755.0437189340591,14.932839602424876,1.188317618562166,6634584358714.0,25950601119411.0]]) #4
valor_2 = np.array([[2,2,2,0.159154943091895,456,377680]]) #0
valor_3 = np.array([[110185.0,1682.6072088479996,25.694668233128425,2.044716730204343,247021319798721.0,72353350887355.0]])
action = loaded_model.predict(valor_2)

print(detect(action))


#%%
#Testing model with all the datasheet for testing
# y_test --> assesment
# x_test --> data for testing

#data_test_model = list(map(detect, x_test))
x_test_1 = np.array([x_test])
list_cases = []
for case in x_test_1:
    #action = loaded_model.predict(case)
    #action = detect(action)
    #print (action)
    #list_cases.append(action)
    print(case)
    print('\n\n')

for case in x_test:
    x_test_n = np.array([case])
    action = loaded_model.predict(x_test_n)
    action = detect(action)
    list_cases.append(action)
    
    
    
x_test_n = np.array([x_test[0]])
action = loaded_model.predict(x_test_n)
print(detect(action))


target_data_train = list(map(str_to_list, y_train))