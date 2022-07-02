# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:14:54 2022

@author: leoes
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix 
import numpy as np
import seaborn as sns

# Read Dataseet
datasheet = pd.read_csv('data/data_for_training_features.csv',engine='python', index_col=0)

datasheet_small = datasheet[(datasheet['assessment'] == 0) | (datasheet['assessment'] == 3) | (datasheet['assessment'] == 5)]
datasheet_big = datasheet[(datasheet['assessment'] == 2) | (datasheet['assessment'] == 4)]


def decision_tree(datasheet):
    # Predictor variable
    x = datasheet.iloc[:,0:6] # takes all columns except "assessment"
    
    # Variable to predict
    y = datasheet.iloc[:,6] # takes just "enfermedad" to predict
    y = y.astype(str) 
    
    y_cases = y.tolist()
    # y_count = y_cases.count("2")    # writte the case [2,4]
    # y_count                         # writte the case [0,3,5]
    
    y_cases = set(y_cases)
    y_cases = sorted(y_cases)
    
    # it is necessary to have:
    #    X_train & y_train for training
    #    Y_test & y_test for testing 
    # train_test_split(data , resutl predict, size of data for training and testing(takes 75% for training and 25% for ), random=0 takes allways de same values)
    x_train, x_test, y_train,y_test = train_test_split(x,y,train_size=0.75, random_state=0)
    #x_train, x_test, y_train,y_test = train_test_split(x,y,train_size=0.80, random_state=1)
    
    # Calling to constuctor of the decisition tree
    # max_depth -> number of leven on the tree
    arbol = DecisionTreeClassifier(max_depth=6) # just take 4 branches of the tree
    #arbol = DecisionTreeClassifier() # makes the complete tree
    
    # Training the model
    arbol_enfermedad = arbol.fit(x_train, y_train)
    
    #plot tree fig
    fig = plt.figure(figsize=(25,20))
    #fig = plt.figure(figsize=(170,160))
    #fig = plt.figure(figsize=(370,360))
    # fig = plt.figure(figsize=(380,370))
    tree.plot_tree(arbol_enfermedad, feature_names=list(x.columns.values),
                    class_names=list(y.values), filled=True)
    plt.show()
    
    # save image 
    #fig.savefig("decision_tree_1.png")
    
    # Predict the response for dataset
    y_pred = arbol_enfermedad.predict(x_test)
    
    
    # Create confution matrix
    matriz_de_confusion = confusion_matrix(y_test, y_pred)

    # crear mapa de calor dibujar mapa de calor
    sns.heatmap(matriz_de_confusion, annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()
    
    return matriz_de_confusion

# process the matrix
matriz_small = decision_tree(datasheet_small)
matriz_big = decision_tree(datasheet_big)

# Calculte efficiency in small data
precision_global_small = np.sum(matriz_small.diagonal()) / np.sum(matriz_small)
precision_0 = ((matriz_small[0,0])) / sum(matriz_small[0,])
precision_3 = ((matriz_small[1,1])) / sum(matriz_small[1,])
precision_5 = ((matriz_small[2,2])) / sum(matriz_small[2,])


# Calculte efficiency in small data
precision_global_big = np.sum(matriz_big.diagonal()) / np.sum(matriz_big)
precision_2 = ((matriz_big[0,0])) / sum(matriz_big[0,])
precision_4 = ((matriz_big[1,1])) / sum(matriz_big[1,])

precision_globaL = (precision_global_small + precision_global_big) / 2



# lista = [
# 0.2608695652173913,
# 0.635593220338983,
# 0.15,
# 0.6625766871165644,
# 0.47368421052631576]

# list_values= sum(lista) / len(lista)


# lista2 = [
#     0.6666666666666666,
#     0.773109243697479,
#     0.7894736842105263,
#     0.8823529411764706,
#     0.7058823529411765
#     ]
# list_values= sum(lista2) / len(lista2)