# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:16:12 2022

@author: Daniel
Tema: Clasificacion lineal
"""
import numpy as np # manejo de matrices
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import math


plt.close()

dt = np.load(r'C:\Users\Daniel\Dropbox\Mi PC (LAPTOP-OVQGSK8G)\Desktop\IA\Codigos en clase\Clasificacion lineal\data_3D.npy', allow_pickle = True).item()

x_a = dt['A']
x_b = dt['B']
X= np.concatenate((x_a,x_b), axis=0)

d= np.size(x_a, axis = 1)
Y= np.concatenate((np.ones((len(x_a),1)),-1*np.ones((len(x_b),1))),axis =0 )

fig = plt.figure()
ax= fig.add_subplot(111, projection = '3d')
ax.scatter(x_a[:,0], x_a[:,1], x_a[:,2], c = 'red' , label = 'clase a')
ax.scatter(x_b[:,0], x_b[:,1], x_b[:,2], c = 'blue' , label = 'clase b')
ax.set_title('Datos')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid()



#division de datos de entrebamiento

traning_a, testing_a = model_selection.train_test_split(x_a,test_size = int(0.3*len(x_a)),train_size = int(0.7*len(x_a)))
traning_b, testing_b = model_selection.train_test_split(x_b,test_size = int(0.3*len(x_b)),train_size = int(0.7*len(x_b)))

traning_matrix = np.concatenate((traning_a,traning_b),axis = 0)
testing_matrix = np.concatenate((testing_a,testing_b),axis = 0)

y_train = np.concatenate((np.ones((len(traning_a),1)),-1*np.ones((len(traning_b),1))),axis =0 )
y_test = np.concatenate((np.ones((len(testing_a),1)),-1*np.ones((len(testing_b),1))),axis =0 )

#-----------------------------------------------algotirmos-----------------------------------------------------

#1) LMS: mimizar el error cuadratico medio.
traning_matrix = np.concatenate((traning_matrix,np.ones((len(traning_matrix),1))),axis = 1)
A= np.matmul(np.transpose(traning_matrix),traning_matrix)
b = np.matmul(np.transpose(traning_matrix),y_train)

if np.linalg.det(A) == 0:
    W = np.matmul(np.linalg.inv(A),b) #(d+1,1){}
else:
    eta = 0.01
    W = np.zeros((d + 1,1))
    ep = 100
    for k in range(ep):
        idx = np.random.permutation(len(traning_matrix))  
        for i in range(len(traning_matrix)):
            h_1 = np.matmul(np.transpose(W),np.transpose(traning_matrix[idx[i],:]))
            e_1 = h_1 - y_train[idx[i]]
            W = W - eta*np.transpose(traning_matrix[idx[i],:]).reshape(d+1,1)*e_1

#--------------------------------------prueba--------------------------------------------------------

testing_matrix = np.concatenate((testing_matrix, np.ones((len(testing_matrix),1))),axis = 1)
y_out = np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix))))

# metricas de rendimiento

c_lms = confusion_matrix(y_test, y_out)
acc_lms = 100*(c_lms[0,0] + c_lms[1,1])/sum(sum(c_lms))
err_lms = 100 - acc_lms
se_lms = 100*c_lms[0,0]/(c_lms[0,0] + c_lms[0,1])
sp_lms = 100*c_lms[1,1]/(c_lms[1,1] + c_lms[1,0])

#2) discrimante logistico 

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1/(1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = 1/(1 + z)
        return sig
        
        
eta = 0.01
W = np.zeros((d + 1,1))
ep = 100
for k in range(ep):
    idx=np.random.permutation(len(traning_matrix))
    
    for i in range(len(traning_matrix)):
        h = np.matmul(np.transpose(W),np.transpose(traning_matrix[idx[i],:]))
        p = y_train[idx[i]]*(sigmoid(y_train[idx[i]]*h))*(traning_matrix[idx[i],:])
        W = np.transpose(np.transpose(W)- eta*p)
        
y_out_LO = -np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix))))
# metricas de rendimiento

c_dl = confusion_matrix(y_test, y_out_LO)
acc_dl = 100*(c_dl[0,0] + c_dl[1,1])/sum(sum(c_dl))
err_dl = 100 - acc_dl
se_dl = 100*c_dl[0,0]/(c_dl[0,0] + c_dl[0,1])
sp_dl = 100*c_dl[1,1]/(c_dl[1,1] + c_dl[1,0])

#3) discirminante fisher (f)
M_a = np.mean(traning_a,axis=0).reshape(1,d) # calculo de la media a
M_b = np.mean(traning_b,axis=0).reshape(1,d) # calculo de la media b

M = M_b - M_a

S_i = np.zeros((d,d)) # inicializamos la matriz de covarianza intra clase

for i in range(int(len(traning_matrix)/2)):
    S_i = S_i + np.matmul(np.transpose(traning_a[i,:] - M_a),traning_a[i,:] - M_a) + \
        np.matmul(np.transpose(traning_b[i,:] - M_b),traning_b[i,:] - M_b) # actualizacion de Si

W = np.matmul(np.linalg.inv(S_i),np.transpose(M)) # obtenemos el vector W a travez de la matriz de covarianza 
#calculo de w_0
M_t = np.mean(traning_matrix[:,:-1], axis = 0).reshape(d,1)  
      
W_0 = -np.matmul(np.transpose(W),M_t)

y_out_f = -np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix[:,:-1])) + W_0))   

c_f = confusion_matrix(y_test, y_out)
acc_f = 100*(c_f[0,0] + c_f[1,1])/sum(sum(c_f))
err_f = 100 - acc_f
se_f = 100*c_f[0,0]/(c_f[0,0] + c_f[0,1])
sp_f = 100*c_f[1,1]/(c_f[1,1] + c_f[1,0])

#4 perceptron

eta = 0.01
W = np.zeros((d + 1,1))
ep = 100
for k in range(ep):
    idx=np.random.permutation(len(traning_matrix))
    
    for i in range(len(traning_matrix)):
        h = np.matmul(np.transpose(W),np.transpose(traning_matrix[idx[i],:]))
        if h*y_train[idx[i]] <= 0:
            W = W + eta*np.transpose(traning_matrix[idx[i],:]).reshape(d+1,1)*y_train[idx[i]]
        
y_out = np.sign(np.transpose(np.matmul(np.transpose(W),np.transpose(testing_matrix))))

c_dp = confusion_matrix(y_test, y_out)
acc_dp = 100*(c_dp[0,0] + c_dp[1,1])/sum(sum(c_dp))
err_dp = 100 - acc_dp
se_dp = 100*c_dp[0,0]/(c_dp[0,0] + c_dp[0,1])
sp_dp = 100*c_dp[1,1]/(c_dp[1,1] + c_dp[1,0])       
