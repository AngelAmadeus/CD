#Practica 1: Calidad de los Datos (Tabla)
import pandas as pd
import numpy as np
#%% Importar los datos de la Base de Datos
# Para importar los datos de los accidentes: El archivo se busca en la carpeta en la que esta el
# documento actual .py ; los .. te mandab una carpeta hacia atras de donde estabas, en seguida ahi 
# buscas Data/Accidents_2015.csv para el archivo de excel
dir_file = '../Data/Accidents_2015.csv'
accidents = pd.read_csv(dir_file, header = 0, sep = ',', index_col = None, skip_blank_lines = True)
#%% Observar Datos
#accidents.head() # muestra los primeros 5 valores de cada variable para observar
#%% lista de variables en la base de datos
columns = pd.DataFrame(list(accidents.columns.values))
#%% Lista de tipos de datos de cada variable
data_types = pd.DataFrame(accidents.dtypes, columns = ['Data_Types'])
#%% Lista de los datos nulos o faltantes
missing_values = pd.DataFrame(accidents.isnull().sum(), columns = ['Missing_Values'])
#%% Lista de la cantidad de datos de cada variable
present_data = pd.DataFrame(accidents.count(), columns = ['Present_Data'])
#%% Lista de valores unicos de cada variable 
unique_values = pd.DataFrame(columns = ['Unique_Values'])
for col in list(accidents.columns.values): unique_values.loc[col] = [accidents[col].nunique()]
#for col in list(accidents.columns.values): unique_values.loc[col] = [accidents[col].unique()]
# si es nunique: da la cantidad de datos unicos; si es unique: da  )un arreglo con los datos unicos)
#%% Lista con los valores minimos de cada variable
minimum_values = pd.DataFrame(columns = ['Minimum_Values'])
for col in list(accidents.columns.values):
    try:
        minimum_values.loc[col] = [accidents[col].min()]
    except:
        pass
#%% Lista con los valores máximos de cada variable
maximum_values = pd.DataFrame(columns = ['Maximum_Values'])
for col in list(accidents.columns.values):
    try:
        maximum_values.loc[col] = [accidents[col].max()]
    except:
        pass
#%% Crear la tabla que muestra la calidad de los datos
data_quality_report = data_types.join(missing_values).join(present_data).join(unique_values).join(minimum_values).join(maximum_values)
