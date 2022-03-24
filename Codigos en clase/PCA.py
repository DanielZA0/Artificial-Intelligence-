# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:17:03 2022

@author: Daniel Zambrano


"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
#-----------------------Importacion de datos-----------------------------------

db = datasets.load_iris()
X = db.data
d = np.size(X,axis=1)#atributos originales
y = db.target

#1. obtener vector de cada caracteisca de cada patron -> X

#2. Se remueve la media de cada uno de los datos.
x_bar=X-np.mean(X, axis = 0)

#3. Cálculo de la matriz de covariza
A = np.transpose(x_bar)
k = np.cov(A)

#4. Calculo de vectores y valores propios de k
eig_values, eig_vec = np.linalg.eig(k)

#5. Conservar m<d valores propios mas grandes y sus respe. vec. 
# Metodo 1 (visialemnte)
plt.figure(dpi = 600)
plt.bar(list(range(d)),eig_values)
plt.title('comportamiento de los valores propios')
plt.xlabel('componenetes')
plt.ylabel('varianza')

plt.grid()
#Metodo 2 (Porcentajes):
# Se tomaran los N componentes que contribuyen el 95% de la varianza de los datos.

part = 100*np.cumsum(eig_values)/sum(eig_values) 
for i in range(len(part)):
    if part[i] >= 95.0:
        m = i + 1
        break
# Metodo 3 visualizacion + porcentajes
part_por_contr = 100*eig_values/sum(eig_values) 

plt.figure(dpi = 600)
plt.bar(list(range(d)),(part_por_contr))
plt.title('porcentaje de contribucion valores propios')
plt.xlabel('componenetes')
plt.ylabel('varianza')
plt.grid()

#6. nueva matriz de datos no correalcionados.
M_pri = eig_vec[:,:2] # matriz de vectores propios reducida (d x m)
A_pri = np.matmul(np.transpose(M_pri),A)
new_X = np.transpose(A_pri)

# representacion grafica de los datos a traves de PCA
plt.figure(dpi = 600)
plt.scatter(new_X[0:49,0], new_X[0:49,1], c = 'red' , label = 'Setosa')
plt.scatter(new_X[50:100,0], new_X[50:100,1], c = 'blue' , label = 'ver')
plt.scatter(new_X[101:150,0], new_X[101:150,1], c = 'green' , label = 'vir')
plt.title('comportamiento de los valores propios')
plt.xlabel('componenetes')
plt.ylabel('varianza')
plt.grid()
