#Practica 5: Variables Dummy
import numpy as np
import pandas as pd
import sklearn.metrics as akm
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
#%% Importación de Datos
data_file = '..\Data\BD_Accidents.csv'
accidents = pd.read_csv(data_file, header = 0)
#%%Reporte de Calidad de Datos
columns = pd.DataFrame(list(accidents.columns.values))
data_types = pd.DataFrame(accidents.dtypes, columns = ['Data_Types'])
missing_values = pd.DataFrame(accidents.isnull().sum(), columns = ['Missing_Values'])
present_data = pd.DataFrame(accidents.count(), columns = ['Present_Data'])
unique_values_qr = pd.DataFrame(columns = ['Unique_Values'])
for col in list(accidents.columns.values): unique_values_qr.loc[col] = [accidents[col].nunique()]
data_quality_report = data_types.join(missing_values).join(present_data).join(unique_values_qr)
#%% Elegir las variables con la menor cantidad de valores unicos
#(las variables deben de ser de tipo int64)
indx = np.array(accidents.dtypes=='int64')
col_list = list(accidents.columns.values[indx])
accidents_int = accidents[col_list]
#%%Elegir los valores unicos mas pequeños de las varialbes enteras
unique_values = pd.DataFrame(columns = ['Unique_Values'])
for col in col_list: unique_values.loc[col] = [accidents_int[col].nunique()]
#Elegir varibles con valores unicos dentro de un umbral
indx = np.array(unique_values['Unique_Values']<=5)
col_list_new = list(accidents_int.columns.values[indx])
#%%Crear variables dummies de "Accidents Severity"
dummy1 = pd.get_dummies(accidents_int[col_list_new[0]],prefix = col_list_new[0])
#%% Crear una tabla que junte las varaibles
accidents_dummy = dummy1
for col in col_list_new[1:]:
    temp = pd.get_dummies(accidents_int[col],prefix=col)
    accidents_dummy = accidents_dummy.join(temp)
#%%Si no existen variables cunatitativas, se obtienen indices de similitud binarios 
dist1 = sc.pdist(accidents_dummy.transpose(),'matching')
dist2 = sc.pdist(accidents_dummy.transpose(),'jaccard')
D1 = sc.squareform(dist1)
D2 = sc.squareform(dist2)
#%%Si existen varaibles cuantitativas, se normalizan y se obtienen los indices de similitud
accidents_dummy_norm = (accidents_dummy - accidents_dummy.mean(axis = 0))/(accidents_dummy.std(axis = 0))
dist3 = sc.pdist(accidents_dummy_norm.transpose(),'euclidean')
D3 = sc.squareform(dist3)
