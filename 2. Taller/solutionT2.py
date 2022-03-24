# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:04:52 2022
@author: Daniel
Solucion de  taller 2
"""

import numpy as np # manejo de matrices
import matplotlib.pyplot as plt # gráficos
from sklearn import model_selection # segmentación de datos (training & testing)
dt=np.load('data.npy',allow_pickle=True)
temp = dt.item(0)

#-------------------------------------------solucion punto dos------------------------------------------------------------------

#extraer primer item de la base de datos

#extraer los datos_3D
data_3d = temp['data_3D']
#extraer las clases de los datos datos_2D

data_3d_a = data_3d['data_a']

data_3d_b = data_3d['data_b']

# separacion de datos 

#1. obtener vector de cada caracteisca de cada patron -> traning_3d_a y b
training_3d_a, testing_3d_a = model_selection.train_test_split(data_3d_a,test_size = int(0.2*len(data_3d_a)),train_size = int(0.8*len(data_3d_a)))
training_3d_b, testing_3d_b = model_selection.train_test_split(data_3d_b,test_size = int(0.2*len(data_3d_b)),train_size = int(0.8*len(data_3d_b)))

d = np.size(training_3d_a,axis=1)
#2. Se remueve la media de cada uno de los datos.
x_bar_a=training_3d_a-np.mean(training_3d_a, axis = 0)
x_bar_b=training_3d_b-np.mean(training_3d_b, axis = 0)

#3. Cálculo de la matriz de covariza
A_a = np.transpose(x_bar_a)
A_b = np.transpose(x_bar_b)

k_a = np.cov(A_a)
k_b = np.cov(A_b)


#4. Calculo de vectores y valores propios de k
eig_values_a, eig_vec_a = np.linalg.eig(k_a)
eig_values_b, eig_vec_b = np.linalg.eig(k_b)


part_por_contr_a = 100*eig_values_a/sum(eig_values_a) 

plt.figure(dpi =1000)
plt.bar(list(range(d)),(part_por_contr_a))
plt.title('porcentaje de contribucion valores propios clase a')
plt.xlabel('características')
plt.ylabel('porcentaje')
plt.grid()

part_por_contr_b = 100*eig_values_a/sum(eig_values_a) 

plt.figure(dpi = 600)
plt.bar(list(range(d)),(part_por_contr_b))
plt.title('porcentaje de contribucion valores propios clase b')
plt.xlabel('características')
plt.ylabel('porcentaje')
plt.grid()

#6. nueva matriz de datos no correalcionados.
M_pri_a = eig_vec_a[:,:2] # matriz de vectores propios reducida (d x m)
A_pri_a = np.matmul(np.transpose(M_pri_a),A_a)
new_Xt_a = np.transpose(A_pri_a)

M_pri_b = eig_vec_b[:,:2] # matriz de vectores propios reducida (d x m)
A_pri_b = np.matmul(np.transpose(M_pri_b),A_b)
new_Xt_b = np.transpose(A_pri_b)

#----------------matriz de entrenamiento-----------------------------
t_a_bar = testing_3d_a - np.mean(training_3d_a, axis = 0)
A_a_t = np.transpose(t_a_bar)
A_t_a_pri = np.matmul(np.transpose(M_pri_a),A_a_t)

testing_new_a = np.transpose(A_t_a_pri) 

t_b_bar = testing_3d_b - np.mean(training_3d_b, axis = 0)
A_b_t = np.transpose(t_b_bar)
A_t_b_pri = np.matmul(np.transpose(M_pri_b),A_a_t)

testing_new_b = np.transpose(A_t_b_pri) 

new_testting_matrix = np.concatenate((testing_new_a,testing_new_b),axis = 0)

# # calculo de matriz de covarianza
k_3d_a= np.cov(np.transpose(new_Xt_a)) 
k_3d_b= np.cov(np.transpose(new_Xt_b))

# #calculo de medias muestrales

u_3d_a = new_Xt_a.mean(axis = 0)
u_3d_b = new_Xt_b.mean(axis = 0)

# #-------------------Implementacion del clasificador bayesiano---------------------------------
 #####estimacion de funcion de verosimilitud
d = np.size(new_Xt_a,axis=1)#atributos originales # numero de atributos
Y_bayes = np.zeros((len(new_testting_matrix),1)) # Arreglo de salida 

ideal = np.concatenate((np.zeros((len(testing_new_a),1)),np.ones((len(testing_new_b),1))),axis = 0) 
# #Calculo de probabilidad
for i in range(len(new_testting_matrix)):
    P_3d_a_1 =(np.exp(-0.5*np.matmul(np.matmul(new_testting_matrix[i,:]-u_3d_a,np.linalg.inv(k_3d_a)),new_testting_matrix[i,:]-u_3d_a)))
    P_3d_b_1 = (np.exp(-0.5*np.matmul(np.matmul(new_testting_matrix[i,:]-u_3d_b,np.linalg.inv(k_3d_b)),new_testting_matrix[i,:]-u_3d_b)))
    
    p_3d_a =  0.5*(1/(np.sqrt(2*(np.pi**d)*np.linalg.det(k_3d_a))))*P_3d_a_1
    p_3d_b =  0.5*(1/(np.sqrt(2*(np.pi**d)*np.linalg.det(k_3d_b))))*P_3d_b_1
    
# #classificacion de los datos de entreno
    P = (p_3d_a,p_3d_b)
    idx = P.index(max(P)) 
#Resultados de cada uno de los datos 

    if idx == 0:
        Y_bayes[i] = 0 #a
        
    if idx == 1:
        Y_bayes[i] = 1 #b
        
a_bayes = []
b_bayes = [] 

for i in range(len(Y_bayes)):
    if Y_bayes[i] == 0: 
        a_bayes.append(new_testting_matrix[i])
    else:
        b_bayes.append(new_testting_matrix[i])
        
a_bayes_arr = np.asarray(a_bayes)
b_bayes_arr = np.asarray(b_bayes)

fig = plt.figure(dpi = 600) 
plt.scatter(a_bayes_arr[:,0],a_bayes_arr[:,1], c = 'grey',label = 'Tipo a')
plt.scatter(b_bayes_arr[:,0],b_bayes_arr[:,1], c = 'orange',label = 'Tipo b')
plt.legend()
plt.grid()
plt.title('Gráfica de tipos')



# #Calculo del error

e_bayes =100*sum(Y_bayes != ideal)/len(Y_bayes)  
print('esto es gausiana:', e_bayes)  
        
    









