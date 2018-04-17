#Practica 7 Algoritmos de Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
#%% Importación de Datos
dir_file = '../Data/BD_Algoritmos_de_clustering.csv'
data = pd.read_csv(dir_file, header = 0)
#%% Aplicar clustering jerarquico
Z = hierarchy.linkage(data, method='complete', metric='euclidean')
#%% Dibujando el dendrograma truncado a 100 hojas
plt.figure(figsize=(8,4))
plt.title('Dendrogram completo')
plt.xlabel('Indice de la muestra')
plt.ylabel('Distancia')
dn = hierarchy.dendrogram(Z,truncate_mode='lastp', p=100)
plt.show()
#%% Criterio del la grafica de codo y el gradiente para determinar el número de grupos
last = Z[-15:,2] #se toman los últimos datos porque son los que tienen las mayores diferencias
last_rev = last[::-1]
idxs = np.arange(1,len(last_rev)+1)
plt.plot(idxs,last_rev)
gradiente = np.diff(last)
grad_rev =gradiente[::-1]
plt.plot(idxs[1:],grad_rev)
plt.title('Grafica de Codo y Criterio de Gradiente')
plt.ylabel('Distancia')
plt.xlabel('Num. Grupos')
plt.grid()
plt.show()
k = grad_rev.argmax()+2
print('Num clusters: ',k)
#%%Grupos
gruposmax = 5 #observar el dendrograma
grupos = hierarchy.fcluster(Z,gruposmax,criterion='maxclust')
pd.value_counts(grupos)
#%% Aplicar el KMeans
model = KMeans(n_clusters=5, init='k-means++')
model = model.fit(data)
#%% Revisar el criterio de grafica de codo
inercia = np.zeros(10)
tiempo = np.zeros(10)
for k in np.arange(1,10):
    model = KMeans(n_clusters=k, init='k-means++')
    model = model.fit(data)
    inercia[k] = model.inertia_
grad = np.diff(inercia)
plt.plot(np.arange(2,10),-1*grad[1:],c='g')
plt.plot(np.arange(1,10),inercia[1:],c='b')
plt.title('Grafica de Codo y Criterio de Gradiente')
plt.ylabel('Distancia')
plt.xlabel('Num. Grupos')
plt.grid()
plt.show()
#%%Visualizar las inercias como cambios porcentuales
inercia_2 = pd.DataFrame(inercia)
inercia_2 = inercia_2.iloc[1:,:]
k = 0
while k <= 7:
    inercia_2.iloc[k,0] = 1-(inercia_2.iloc[k+1,0]/inercia_2.iloc[k,0])
    k = k+1
inercia_2 = inercia_2.iloc[0:8,:]
#%% Clasificar los datos segun el numero de grupos determinado
model = KMeans(n_clusters=5, init='k-means++')
model = model.fit(data)
Ypredict = model.predict(data)
pd.value_counts(Ypredict)
#%% Visualizar los resultados
indx = Ypredict==1
grupo1 = data.iloc[indx,:]
plt.figure(figsize=(6,5))
plt.plot(np.array(grupo1.transpose()))
plt.plot(model.cluster_centers_[1],c='k')
plt.title('Base de Datos')
plt.ylabel('Valor')
plt.xlabel('Variable')
plt.show()
#Esto muestras los datos en x dimensiones, siendo cada valor en x una dimensión
#La raya negra muetra el promedio de los datos
#%%Visualización del comportamiento medio de los grupos en sus 30 dimensiones
plt.plot(model.cluster_centers_.transpose())
plt.title('Comportamiento Promedio de Cada Cluster')
plt.ylabel('Valor')
plt.xlabel('Variable')
plt.show()
