# Practica 15: Support Vector Machine
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
#%%Crear los datos a clasificar
np.random.seed(0)
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]] #Hace 20 puntitos de 2 valores aleatorios, con dist normal, de media 0 y desvest igual a 1, centrados en (-2,-2) y (2,2)
Y = [0]*20+[1]*20
#%%Crear el modelo SVM:lineal
#SVC es para problemas de clasificación
#SVR es para problemas de regresión
clf = svm.SVC(kernel = 'linear',)#clasificador
#el kernel es una tranformación de dimensión, es decir que un puntito que se graficaba en
# R^2 [x1,x2] se transforma el puntito en R^5[x1,x2,x3,x4,x5], todo esto, porque en la otra
# dimension puede existir un hiperplano que los separe.
#KERNEL LINEAL('linear'): R^2 a R^2 donde V = wo + wix1 + w2x2, y R^3 a R^3 y R^30 a R^30
#KERNEL POLINOMIAL('poly'): R^2 a polinomio de grado n a R^m V = wo + w1x1 + w2x2 + w3x1^2 + w4x2^2 + w5x1x2
#KERNEL GAUSSIANO('rbf'): R^2 a (1/((2*pi*sigma)^1/2))*e^(-((x-miu)^2)/(sigma^2)) a R^m; V = wo + w1G1(miu1,sigma1) + w2G2(miu2,sigma2) + .... + wmGm(mium,sigmam)
#KERNEL SIGMOIDAL(''): V = wo + w1S1(alpha1,beta1) + w2S2(alpha2,beta2) + .... + wmSm(alpham,betam)
clf.fit(X,Y)
#%%Crear el hiperplano de separación
W = clf.coef_[0]
a = -W[0]/W[1]
xx = np.linspace(-5,5)
yy = a*xx+(-clf.intercept_[0]/W[1])#Dibihar la intercepcion del hiperplano en los ejes x1 x2
#%%Graficar
plt.plot(xx,yy,'k')
plt.scatter(X[:,0],X[:,1], c=Y)
#Nota: con "clf.support_vectors_" en la terminal de python, se pueden observar las coordenadas de los vectores soporte
#Es decir de los puntos más cercanos a la raya, los puntos con los que se hace el analisis computacional
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1], facecolor = 'red')
#Graficar las lineas de los vectores soporte
b = clf.support_vectors_[0]
yy_0 = a*xx+(b[1]-a*b[0])
b = clf.support_vectors_[-1]
yy_1 = a*xx+(b[1]-a*b[0])
plt.plot(xx,yy_0,'k--')
plt.plot(xx,yy_1,'k--')
plt.grid()
plt.show()
#%%