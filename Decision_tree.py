# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 12:34:06 2022

@author: leoes

@Desition Tree
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


# Predictor variable
x = datasheet.iloc[:,0:6] # takes all columns except "assessment"

# Variable to predict
y = datasheet.iloc[:,6] # takes just "enfermedad" to predict
y = y.astype(str) 

y_cases = y.tolist()

y_count = y.count("0")

y_cases = set(y_cases)
y_cases = sorted(y_cases)

# it is necessary to have:
#    X_train & y_train for training
#    Y_test & y_test for testing 
# train_test_split(data , resutl predict, size of data for training and testing(takes 75% for training and 25% for ), random=0 takes allways de same values)
x_train, x_test, y_train,y_test = train_test_split(x,y,train_size=0.75, random_state=0)

# Calling to constuctor of the decisition tree
# max_depth -> number of leven on the tree
arbol = DecisionTreeClassifier() # just take 4 branches of the tree
#arbol = DecisionTreeClassifier() # makes the complete tree

# Training the model
arbol_enfermedad = arbol.fit(x_train, y_train)

#plot tree fig
#fig = plt.figure(figsize=(25,20))
#fig = plt.figure(figsize=(170,160))
#fig = plt.figure(figsize=(370,360))
fig = plt.figure(figsize=(380,370))
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


# # Prediction global
precision_global = np.sum(matriz_de_confusion.diagonal()) / np.sum(matriz_de_confusion)

# Precisión for 
precision_0 = ((matriz_de_confusion[0,0])) / sum(matriz_de_confusion[0,])
precision_1 = ((matriz_de_confusion[1,1])) / sum(matriz_de_confusion[1,])
precision_2 = ((matriz_de_confusion[2,2])) / sum(matriz_de_confusion[2,])
precision_3 = ((matriz_de_confusion[3,3])) / sum(matriz_de_confusion[3,])
precision_4 = ((matriz_de_confusion[4,4])) / sum(matriz_de_confusion[4,])
# # Precision for every class does not have illness
# precision_NO = ((matriz_de_confusion[0,0])) / sum(matriz_de_confusion[0,])

# # Precision for every class has the illnesss
# precision_SI = ((matriz_de_confusion[1,1])) / sum(matriz_de_confusion[1,])







