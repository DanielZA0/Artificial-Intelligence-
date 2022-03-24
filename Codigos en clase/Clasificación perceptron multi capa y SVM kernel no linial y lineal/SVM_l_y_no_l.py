# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:21:54 2022

@author: Daniel

Solucion de clasificacion no lineal 
"""

import numpy as np # manejo de matrices
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
print('\014')

# importacion de caracteristicas
traning_matrix = np.loadtxt(r'C:\Users\Daniel\Dropbox\Mi PC (LAPTOP-OVQGSK8G)\Desktop\IA\Codigos en clase\Clasificación perceptron multi capa y SVM kernel no linial y lineal\Data\X_train.txt')
testing_matrix = np.loadtxt(r'C:\Users\Daniel\Dropbox\Mi PC (LAPTOP-OVQGSK8G)\Desktop\IA\Codigos en clase\Clasificación perceptron multi capa y SVM kernel no linial y lineal\Data\X_test.txt')

#importacion de etiquetas
Y_train = np.loadtxt(r'C:\Users\Daniel\Dropbox\Mi PC (LAPTOP-OVQGSK8G)\Desktop\IA\Codigos en clase\Clasificación perceptron multi capa y SVM kernel no linial y lineal\Data\y_train.txt')
Y_test = np.loadtxt(r'C:\Users\Daniel\Dropbox\Mi PC (LAPTOP-OVQGSK8G)\Desktop\IA\Codigos en clase\Clasificación perceptron multi capa y SVM kernel no linial y lineal\Data\y_test.txt')

# extraccion de datos de validacion (15% de los de entrenamiento)
#### verificacion de balance de clases

y_train_1 = len (np.where(Y_train == 1)[0]) 
y_train_2 = len (np.where(Y_train == 2)[0]) 
y_train_3 = len (np.where(Y_train == 3)[0]) 
y_train_4 = len (np.where(Y_train == 4)[0]) 
y_train_5 = len (np.where(Y_train == 5)[0]) 
y_train_6 = len (np.where(Y_train == 6)[0]) 

#segmentacion de patrones segun su clase
#se genera matiz de entrenamiento por cada una de las caracteristicas
traning_matrix_1 = traning_matrix[np.where(Y_train == 1)[0],:]
traning_matrix_2 = traning_matrix[np.where(Y_train == 2)[0],:]
traning_matrix_3 = traning_matrix[np.where(Y_train == 3)[0],:]
traning_matrix_4 = traning_matrix[np.where(Y_train == 4)[0],:]
traning_matrix_5 = traning_matrix[np.where(Y_train == 5)[0],:]
traning_matrix_6 = traning_matrix[np.where(Y_train == 6)[0],:]

# division de datos de entrenamiento y validacion
train_1, val_1 = model_selection.train_test_split(traning_matrix_1, test_size= int(0.15*len(traning_matrix_1)), train_size = int(0.85*len(traning_matrix_1)))

train_2, val_2 = model_selection.train_test_split(traning_matrix_2, test_size= int(0.15*len(traning_matrix_2)), train_size = int(0.85*len(traning_matrix_2)))

train_3, val_3 = model_selection.train_test_split(traning_matrix_3, test_size= int(0.15*len(traning_matrix_3)), train_size = int(0.85*len(traning_matrix_3)))

train_4, val_4 = model_selection.train_test_split(traning_matrix_4, test_size= int(0.15*len(traning_matrix_4)), train_size = int(0.85*len(traning_matrix_4)))

train_5, val_5 = model_selection.train_test_split(traning_matrix_5, test_size= int(0.15*len(traning_matrix_5)), train_size = int(0.85*len(traning_matrix_5)))

train_6, val_6 = model_selection.train_test_split(traning_matrix_6, test_size= int(0.15*len(traning_matrix_6)), train_size = int(0.85*len(traning_matrix_6)))

# concatenacion de matrices 

tratraning_matrix = np.concatenate((train_1,train_2,train_3,train_4,train_5,train_6), axis = 0)
valid_matrix = np.concatenate((val_1,val_2,val_3,val_4,val_5,val_6), axis = 0)

y_train = np.concatenate((np.ones(len(train_1)), 2*np.ones(len(train_2)), 3*np.ones(len(train_3)) , 4*np.ones(len(train_4)) , 5*np.ones(len(train_5)), 6*np.ones(len(train_6))), axis = 0)

y_val = np.concatenate((np.ones(len(val_1)), 2*np.ones(len(val_2)), 3*np.ones(len(val_3)) , 4*np.ones(len(val_4)) , 5*np.ones(len(val_5)), 6*np.ones(len(val_6))), axis = 0)

#conversion a variables de dumies
y_train_categorical = pd.get_dummies(y_train)
y_val_categorical = pd.get_dummies(y_val)

d = np.size(traning_matrix, axis = 1)

## -------------------------elavoracion de codigos-----------------------------
##----------------------Perceptron multicapa(ANN - DNN)------------------------
#contruccion de la DNN
dnn_model = Sequential()
dnn_model.add(Dense(d, activation = 'sigmoid', input_shape = (d,) )) # capa de entrada
dnn_model.add(Dropout(0.2))
# capa oculta
dnn_model.add(Dense(16, activation = 'sigmoid'))
dnn_model.add(Dropout(0.1))
dnn_model.add(Dense(8, activation = 'sigmoid'))
dnn_model.add(Dropout(0.1))

# capa de salida
dnn_model.add(Dense(np.size(y_train_categorical, axis = 1), activation = 'softmax'))

# Configuracion de hiperparametros adicionales 
dnn_model.compile(optimizer = 'adam', 
                  loss='categorical_crossentropy',
                  metrics = 'categorical_accuracy')
#Fase de entrenamiento 
dnn_model.fit(tratraning_matrix,
              y_train_categorical,
              epochs = 10,
              verbose = 1,
              workers = 1,
              use_multiprocessing = True,
              validation_data=(valid_matrix, y_val_categorical))
# Prueba:
y_hat = dnn_model.predict(testing_matrix)
y_out = y_hat.round()

# Transformación de datos: 
y_out = pd.DataFrame(y_out)
y_out = y_out.values.argmax(1) + 1

# Métricas de evaluación: 
c_dnn = confusion_matrix(Y_test, y_out)