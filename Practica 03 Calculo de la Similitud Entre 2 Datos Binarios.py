#Practica 3: Calculo de la similitud entre 2 datos binarios
import numpy as np
import pandas as pd
from sklearn import datasets
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
#%% Importar o cargar datos
digits = datasets.load_digits()
#%% Visualización de una muestra de los datos
ndig = 20 #es el tamaño de la muestra que se va a tomar
for k in np.arange(ndig):
    plt.subplot(2,ndig/2,k+1)   # hace una matriz de plots o de imagenes
    plt.axis('off')             # 
    plt.imshow(digits.images[k],cmap=plt.cm.gray_r) # muestra las imagenes "k" de digits y lo pone en escala de grises
    plt.title('Digit: %i'%k)
#%% Cambiar la forma de mis datos a forma binaria
data = digits.data[0:20]
umbral = 7
data[data<=umbral] = 0
data[data>umbral] = 1
ndig = 10 

for k in np.arange(ndig):
    plt.subplot(2,ndig/2,k+1)   # hace una matriz de plots o de imagenes
    plt.axis('off')             # 
    plt.imshow(np.reshape(data[k],(8,8)),cmap=plt.cm.gray_r) # las filas de datas acomoda esa fila como una matriz de 8 por 8 con el reshape
    plt.title('Digit: %i'%k)
#%% Calculo del indice de similitud
#matriz de confución
cf_m = skm.confusion_matrix(data[0],data[1])
#indice de emparejamiento simple
sim_simple = skm.accuracy_score(data[0],data[1])
sim_simple1 = (cf_m[0,0]+cf_m[1,1]/np.sum(cf_m))
#indice de Jaccard ( No sirve, está mal programada )
sim_jac = skm.jaccard_similarity_score(data[0],data[1])
#indice de Jaccard corregido
sim_jac_real = cf_m[1,1]/(np.sum(cf_m)-cf_m[0,0])
#comparacion del 0 con el 10 que en realidad es la comparación entre 2 ceros
sim_simple2 = skm.accuracy_score(data[0],data[10])
cf_m2 = skm.confusion_matrix(data[0],data[10])
sim_jac_real2 = cf_m2[1,1]/(np.sum(cf_m2)-cf_m2[0,0])
#%% Calculo de las distancias según scipy
# emparejamiento simple
dist1 = sc.matching(data[0],data[1])
# distancia de Jaccard
dist1j = sc.jaccard(data[0],data[1])
# distancia de todos contra todos
dsitt1 = sc.pdist(data,'matching')
dsitt1 = sc.squareform(dsitt1)
dsitt1j = sc.pdist(data,'jaccard')
dsitt1j = sc.squareform(dsitt1j)
