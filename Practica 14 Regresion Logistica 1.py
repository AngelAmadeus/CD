#Practica 14: Regresión Logística 1
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
#%% Importación de Datos
data_file = '../Data/ex2data1.txt'
data = pd.read_csv(data_file,header=None)
#%%Visualización de Datos
X = data.iloc[:,0:2]
Y = data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
#%%Regresión Logística
def reg_log(W,X,Y):
    V = np.matrix(X)*np.matrix(W).transpose()
    return np.array(1/(1+np.exp(-V)))[:,0]
#%%Función de Costo
def fun_cost(W,X,Y):
    Y_estimada = reg_log(W,X,Y)
    return np.sum(-Y*np.log(Y_estimada)-(1-Y)*np.log(1-Y_estimada))/len(Y)
#%%Inicializar Variables
poly = PolynomialFeatures(1) #un polinomio de primer orden, es decir con exponentes de 1
X_asterisco = poly.fit_transform(X) #Agrega una fila de unos a la variable X 
W = np.zeros(3)
Resultados_1 = opt.minimize(fun_cost,W,args =(X_asterisco,Y))
W = Resultados_1.x #el x es la variable que inimiza internamente opt, es el nombre interno
#%%Interpretación
x1=np.arange(30,100,0.05)
x2=np.arange(30,100,0.05)
#Todos los puntos posibles
X1,X2=np.meshgrid(x1,x2)
m,n=np.shape(X1)
X1r=np.reshape(X1,(m*n,1))
m,n=np.shape(X2)
X2r=np.reshape(X2,(m*n,1))
#%%Nueva tabla
Xnew=np.append(X1r,X2r, axis=1) #axis=1 las pega por filas, hace dos columnas
Xa_new = poly.fit_transform(Xnew)
Yg=reg_log(W,Xa_new,X1r)
#%% Acomodarlo a su forma original
Z=np.reshape(Yg,(m,n))
Z=np.round(Z)
#Dibujar la superficie
plt.contour(X1,X2,Z)
plt.scatter(X[0],X[1],c=Y)
plt.show()