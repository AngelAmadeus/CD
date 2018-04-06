#Practica 4: Indices de Similitud Cuantitativos
import numpy as np
import pandas as pd
import sklearn.metrics as akm
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
#%% Importación de Datos
data_file = '..\Data\BD_Contaminacion.xlsx'
#para abrir una pestaña en particular
data_Atemajac = pd.read_excel(data_file, header = 0, sheetname='Atemajac')
#%%Borrar todas las filas que tengan "nan"
#data_Atemajac = data_Atemajac.dropna()
#%%Borrar todos los "nan" en [fila:fila, columna:columna]
data_Atemajac = data_Atemajac.iloc[:,0:7].dropna()
#%%
plt.scatter(data_Atemajac.iloc[:,2],data_Atemajac.iloc[:,3])
#plt.scatter(data_Atemajac['CO'],data_Atemajac['NO2'])
#plt.scatter(data_Atemajac.CO,data_Atemajac.NO2)
plt.xlabel('CO')
plt.ylabel('NO2')
#pone la misma escala
plt.axis('square')
plt.show()
#
D1 = sc.squareform(sc.pdist(data_Atemajac.iloc[:,2:7],'euclidean')) #Compara filas con filas
D2 = sc.squareform(sc.pdist(data_Atemajac.iloc[:,2:7].transpose(),'euclidean')) #Compara columnas con columnas
#por la desigualdad de las escalas, se tiene que hacer una normalización
data_Atemajac_norm = (data_Atemajac.iloc[:,2:7] - data_Atemajac.iloc[:,2:7].mean(axis=0))/data_Atemajac.iloc[:,2:7].std(axis=0)
D1_norm = sc.squareform(sc.pdist(data_Atemajac_norm,'euclidean')) #Compara filas con filas
D2_norm = sc.squareform(sc.pdist(data_Atemajac_norm.transpose(),'euclidean')) #Compara columnas con columnas
#axis 0 por columna
#axis 1 por fila
