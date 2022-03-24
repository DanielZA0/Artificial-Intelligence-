# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:04:52 2022

@author: Daniel
SOlucion de  taller 1
"""
import numpy as np # manejo de matrices
import matplotlib.pyplot as plt # gráficos
from sklearn import model_selection # segmentación de datos (training & testing)
dt=np.load('data.npy',allow_pickle=True)
temp = dt.item(0)

 #------------------------------solucion punto uno-----------------------------------------------------



#extraer primer item de la base de datos
temp = dt.item(0)
#extraer los datos_2D
data_2d = temp['data_2D']
#extraer las clases de los datos datos_2D

data_2d_a = data_2d['data_a']

data_2d_b = data_2d['data_b']

#grafica de datos
fig0 = plt.figure() 
plt.scatter(data_2d_a[1],data_2d_a[0], c = 'grey',label = 'tipo a')
plt.scatter(data_2d_b[1],data_2d_b[0], c = 'orange',label = 'tipo b')
plt.legend()
plt.grid()
plt.title('gráfica de tipos')


# Calculo del centro de cada clase
u_a = data_2d_a.mean(axis = 1)
u_b = u_a = data_2d_b.mean(axis = 1)

# Calculo de la matriz de covarianza
K_2d_a = np.cov(data_2d_a) 
K_2d_b = np.cov(data_2d_b) 

#grafica de histogramas por cada atributo
##Clase a




x_a = np.array(data_2d_b[0])   #turn x,y data into numpy arrays
y_a = np.array(data_2d_b[1])

fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')

#make histogram stuff - set bins - I choose 20x20 because I have a lot of data
hist, xedges, yedges = np.histogram2d(x_a, y_a, bins=(20,20))
xpos_a, ypos_a = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos_a = xpos_a.flatten()/2.
ypos_a = ypos_a.flatten()/2.
zpos_a = np.zeros_like (xpos_a)

dx_a = xedges [1] - xedges [0]
dy_a = yedges [1] - yedges [0]
dz_a = hist.flatten()

ax.bar3d(xpos_a, ypos_a, zpos_a, dx_a, dy_a, dz_a, zsort='average')
plt.title("Datos 2D tipo b")
plt.xlabel("Caracteristica 1")
plt.ylabel("Caracteristica 2")
plt.show()


#-------------------------------------------solucion punto dos------------------------------------------------------------------

#extraer primer item de la base de datos

#extraer los datos_3D
data_3d = temp['data_3D']
#extraer las clases de los datos datos_2D

data_3d_a = data_3d['data_a']

data_3d_b = data_3d['data_b']

# separacion de datos 
training_3d_a, testing_3d_a =model_selection.train_test_split(data_3d_a,test_size = int(0.2*len(data_3d_a)),train_size = int(0.8*len(data_3d_a)))
training_3d_b, testing_3d_b =model_selection.train_test_split(data_3d_b,test_size = int(0.2*len(data_3d_b)),train_size = int(0.8*len(data_3d_b)))

testing_matrix = np.concatenate((testing_3d_a,testing_3d_b),axis = 0) 

#grafica de los datos de entrenamiento.
fig = plt.figure()

fig_3d_trainig=fig.add_subplot(111, projection='3d')

fig_3d_trainig.scatter(training_3d_a[:,0],training_3d_a[:,1],training_3d_a[:,2],c = 'blue',label = 'Entrenamiento_3d_a')
fig_3d_trainig.scatter(training_3d_b[:,0],training_3d_b[:,1],training_3d_b[:,2],c = 'green',label = 'Entrenamiento_3d_b')
fig_3d_trainig.view_init(10, 50)
plt.title("Datos 3D")
plt.xlabel("Caracteristica 1")
plt.ylabel("Caracteristica 2")
fig_3d_trainig.set_zlabel("Caracteristica 3")
plt.show()

# calculo de matiz de covarianza
k_3d_a= np.cov(np.transpose(training_3d_a)) 
k_3d_b= np.cov(np.transpose(training_3d_b))

#calculo de medias muestrales

u_3d_a = training_3d_a.mean(axis = 0)
u_3d_b = training_3d_b.mean(axis = 0)

#-------------------Implementacion del clasificador bayesiano---------------------------------
#####estimacion de funcion de verosimilitud
n = 3 # numero de atributos
Y_bayes = np.zeros((len(testing_matrix),1)) # Arreglo de salida 

ideal = np.concatenate((np.zeros((len(testing_3d_a),1)),np.ones((len(testing_3d_b),1))),axis = 0) 
#Calculo de probabilidad
for i in range(len(testing_matrix)):
    P_3d_a_1 =(np.exp(-0.5*np.matmul(np.matmul(testing_matrix[i,:]-u_3d_a,np.linalg.inv(k_3d_a)),testing_matrix[i,:]-u_3d_a)))
    P_3d_b_1 = (np.exp(-0.5*np.matmul(np.matmul(testing_matrix[i,:]-u_3d_b,np.linalg.inv(k_3d_b)),testing_matrix[i,:]-u_3d_b)))
    
    p_3d_a =  0.5*(1/(np.sqrt(2*(np.pi*n)*np.linalg.det(k_3d_a))))*P_3d_a_1
    p_3d_b =  0.5*(1/(np.sqrt(2*(np.pi**n)*np.linalg.det(k_3d_b))))*P_3d_b_1
    
#classificacion de los datos de entreno
    P = (p_3d_a,p_3d_b)
    idx = P.index(max(P)) 
#REsultados de cada uno de los datos 

    if idx == 0:
        Y_bayes[i] = 0 #a
        
    if idx == 1:
        Y_bayes[i] = 1 #b
        
fig3 = plt.figure() 
ppp=plt.hist(Y_bayes)
plt.title("Resultado Gaussiano")
plt.xlabel("clase a                     clase b")
plt.grid() 


#Calculo del error
e_bayes =100*sum(Y_bayes != ideal)/len(Y_bayes)

#---------------------Clasificador bayesiano naive-----------------------------
#Calculo de las desviación estandar.
o_3d_a = training_3d_a.std(axis=0)
o_3d_b = training_3d_b.std(axis=0)

#construccion de funcion de verosimilitud
Y_bayes_n = np.zeros((len(testing_matrix),1))

for i in range(len(testing_matrix)):
    p_3d_a_n = []
    p_3d_b_n = []
    for j in range(n):
        p_3d_a_n.append(0.5*(1/(np.sqrt(2*(np.pi**n))*o_3d_a[j]))*np.exp(-0.5*(testing_matrix[i,j]-u_3d_a)**2/o_3d_a[j]**2))
        p_3d_b_n.append(0.5*(1/(np.sqrt(2*(np.pi**n))*o_3d_b[j]))*np.exp(-0.5*(testing_matrix[i,j]-u_3d_b)**2/o_3d_b[j]**2))
        
    p_3d_a_n = np.prod(p_3d_a_n)
    p_3d_b_n = np.prod(p_3d_b_n)
    
    #classificacion de los datos de entreno
    P_n = (p_3d_a_n,p_3d_b_n)
    idx_n = P_n.index(max(P_n)) 
    #REsultados de cada uno de los datos 

    if idx_n == 0:
            Y_bayes_n[i] = 0 #a
            
    if idx_n == 1:
            Y_bayes_n[i] = 1 #b
            
#Histograma de 
fig4 = plt.figure() 
ppp=plt.hist(Y_bayes)
plt.title("Resultado Naive")
plt.xlabel("clase a                        clase b")   
plt.grid() 

e_bayes_n =100*sum(Y_bayes_n != ideal)/len(Y_bayes_n)
print('esto es naive:', e_bayes_n)      
print('esto es gausiana:', e_bayes)  
        
    









