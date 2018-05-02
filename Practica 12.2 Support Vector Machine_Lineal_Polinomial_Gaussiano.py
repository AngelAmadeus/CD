# Practica 12.2: Support Vector Machine
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
#%%Importación de datos
data=pd.read_csv('../Data/BD_LogReg_SVM.txt',header=None)
X=data.iloc[:,0:2]
Y=data.iloc[:,2]
#%%Crear modelo de SVM: Lineal
#SVC es para problemas de clasificación
#SVR es para problemas de regresión
clf_1 = svm.SVC(kernel = 'linear')#clasificador
clf_1.fit(X,Y)#
#%%Dibujar la frontera de separación con los puntos (opcional, dependiendo de el númer de variables)
x_min,x_max = X[0].min()-0.5,X[0].max()+0.5
y_min,y_max = X[1].min()-0.5,X[1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()]) #Aqui estan todas las coordenadas de la malla
Z = clf_1.predict(Xnew)#Superficie
Z = Z.reshape(xx.shape)#Esto de la altura de la separación pero del mismo tamaño que xx
plt.figure(1)
plt.contour(xx,yy,Z, cm = plt.cm.Paired)
plt.scatter(X[0],X[1],c = Y, edgecolor = 'k', cmap = plt.cm.Paired)
plt.title('Cúmulo de Datos con Frontera Lineal')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()
#%%Evaluación del modelo
Y_estimado = clf_1.predict(X)
print("\tAccuracy: %1.3f"%accuracy_score(Y,Y_estimado)) #Se escribe g en lugar de f para que sea en cifras significativas
print("\tPrecision: %1.3f"%precision_score(Y,Y_estimado))
print("\tRecall: %1.3f"%recall_score(Y,Y_estimado))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%Crear modelo de SVM: Polinomial
#SVC es para problemas de clasificación
#SVR es para problemas de regresión
clf_2 = svm.SVC(kernel = 'poly', degree=2)#clasificador
clf_2.fit(X,Y)#
#%%Dibujar la frontera de separación con los puntos (opcional, dependiendo de el númer de variables)
x_min,x_max = X[0].min()-0.5,X[0].max()+0.5
y_min,y_max = X[1].min()-0.5,X[1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()]) #Aqui estan todas las coordenadas de la malla
Z = clf_2.predict(Xnew)#Superficie
Z = Z.reshape(xx.shape)#esto de la altura de la separación pero del mismo tamaño que xx
plt.figure(1)
plt.contour(xx,yy,Z, cm = plt.cm.Paired)
plt.scatter(X[0],X[1],c = Y, edgecolor = 'k', cmap = plt.cm.Paired)
plt.title('Cúmulo de Datos con Frontera Polinomial')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()
#%%Evaluación del modelo
Y_estimado = clf_2.predict(X)
print("\tAccuracy: %1.3f"%accuracy_score(Y,Y_estimado)) #Se escribe g en lugar de f para que sea en cifras significativas
print("\tPrecision: %1.3f"%precision_score(Y,Y_estimado))
print("\tRecall: %1.3f"%recall_score(Y,Y_estimado))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%Crear modelo de SVM: Gaussiano
#SVC es para problemas de clasificación
#SVR es para problemas de regresión
clf_3 = svm.SVC(kernel = 'rbf', gamma = 10)#clasificador, gamma por default es 1/num_variables, es como la densidad de gaussianas en el espacio
clf_3.fit(X,Y)#
#%%Dibujar la frontera de separación con los puntos (opcional, dependiendo de el númer de variables)
x_min,x_max = X[0].min()-0.5,X[0].max()+0.5
y_min,y_max = X[1].min()-0.5,X[1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()]) #Aqui estan todas las coordenadas de la malla
Z = clf_3.predict(Xnew)#Superficie
Z = Z.reshape(xx.shape)#esto de la altura de la separación pero del mismo tamaño que xx
plt.figure(1)
plt.contour(xx,yy,Z, cm = plt.cm.Paired)
plt.scatter(X[0],X[1],c = Y, edgecolor = 'k', cmap = plt.cm.Paired)
plt.title('Cúmulo de Datos con Frontera Gaussiana')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()
#%%Evaluación del modelo
Y_estimado = clf_3.predict(X)
print("\tAccuracy: %1.3f"%accuracy_score(Y,Y_estimado)) #Se escribe g en lugar de f para que sea en cifras significativas
print("\tPrecision: %1.3f"%precision_score(Y,Y_estimado))
print("\tRecall: %1.3f"%recall_score(Y,Y_estimado))
