#Practica 11: Compresion de Datos - Principal Component Analisis
import numpy as np
import matplotlib.pyplot as plt
#%%Proponer datos
data = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],
                 [2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]])
#%%Graficar
plt.scatter(data[:,0],data[:,1])
plt.grid()
plt.show()
#%%Convertir el Conjunto de Datos a media cero
#Nota, media cero y estandarizar no es lo mismo!
medias = data.mean(axis=0)
data_m = data-medias
plt.scatter(data_m[:,0],data_m[:,1])
plt.grid()
plt.show()
#%%Calcular la matriz de covarianzas
data_cov = np.cov(data_m.transpose())
#%%Calcular los eigenvectores y eigenvalores
val_prop,vec_prop = np.linalg.eig(data_cov)
#Los eigenvectores: Direccion de la dispersion (lineas punteadas a continuacion)
#Los eigenvalores: Importancia de las Direcciones
#%%Dibujar las Direciones de eigenvectores
#En la grifica los datos estan entre -2 y 2
x = np.arange(-2,2,0.05)
plt.scatter(data_m[:,0],data_m[:,1])
plt.plot(x,(vec_prop[1,0]/vec_prop[0,0])*x, 'm--')
plt.plot(x,(vec_prop[1,1]/vec_prop[0,1])*x, 'g--')
plt.axis('square')
plt.grid()
plt.show()
#las lineas deben de ser ortogonales
#%%Transformas los Datos
componentes = val_prop[[1,0]] #ordenar de mayor a menor
transform = vec_prop[:,[1,0]]
data_new = np.matrix(data_m)*np.matrix(transform)
data_new = np.array(data_new)
#%%Graficar la nueva informacion
x = np.arange(-2,2,0.05)
plt.subplot(121)
plt.scatter(data_m[:,0],data_m[:,1])
plt.plot(x,(vec_prop[1,0]/vec_prop[0,0])*x, 'm--')
plt.plot(x,(vec_prop[1,1]/vec_prop[0,1])*x, 'g--')
plt.axis('square')
plt.subplot(122)
plt.scatter(data_new[:,0],data_new[:,1], c = 'r')
plt.axis('square')
plt.grid()
plt.show()
#%%Reducir la Dimension de los Datos
componentes = val_prop[[1]]
transform = np.reshape(vec_prop[:,1],[2,1])

data_new = np.matrix(data_m)*np.matrix(transform)
data_new = np.array(data_new)
