#Pratica 2: Consultas de la Base de Datos e Histogramas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% Importación de Datos
dir_file = '../Data/Accidents_2015.csv'
accidents = pd.read_csv(dir_file, header = 0, sep = ',', index_col = None, skip_blank_lines = True)
#%% Reportes predeterminados de Pandas
quick_report1 = pd.DataFrame(accidents.describe().transpose())
#para que incluya los datos tipo "object"
quick_report2 = pd.DataFrame(accidents.describe(include=['object']).transpose())
#%% Consultas básicas
#consultas de accidentes por cada día (ambos son iguales, pero por diferente método)
Acc_by_date = pd.DataFrame(pd.value_counts(accidents['Date']))
Acc_by_date2 = pd.DataFrame(accidents.groupby(['Date'])['Date'].count())
#consulta de número de vehículos que chocaron al día
Veh_by_date = pd.DataFrame(accidents.groupby(['Date'])['Number_of_Vehicles'].sum())
#consulta de número de heridos que chocaron al día
Cas_by_date = pd.DataFrame(accidents.groupby(['Date'])['Number_of_Casualties'].sum())
#%% Juntando las tablas anteriores
quick_report3 = Acc_by_date.join(Veh_by_date).join(Cas_by_date)
#consulta de vehiculos involucrados en los accidentes en cada minuto del día
time_count = pd.DataFrame(accidents.groupby(['Time'])['Number_of_Vehicles'].sum())
#consulta de vehiculos involucrados en los accidentes en fecha y minuto
date_time_count = pd.DataFrame(accidents.groupby(['Date','Time'])['Number_of_Vehicles'].sum())
#%% Juntar Tablas (otro método) (Busca que las tablas que se juntan, tengan el mismo índice)
vec_casualties = Cas_by_date.merge(Veh_by_date,left_index= True,right_index = True)
#%% Realizar gráficas de los datos
# Crear histogramas (1)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24)
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram')
plt.show()
# Crear histogramas (2)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,normed=True)
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Probability')
plt.title('Vehicles Histogram (Normed')
plt.show()
# Crear histogramas (3)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,normed=True,cumulative=True)
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Probability')
plt.title('Vehicles Histogram (Cumulative Normed)')
plt.show()
# Crear histogramas (4)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,cumulative=True)
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram (Cumulative)')
plt.show()
# Crear histogramas (5)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,histtype = 'step' )
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram')
plt.show()
# Crear histogramas (6)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,histtype = 'stepfilled' )
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram')
plt.show()
# Crear histogramas (7)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,histtype = 'stepfilled',color = 'r' )
plt.xlabel('Number of Vehicles (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram')
plt.show()
# Crear histogramas (8)
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,histtype = 'step',color = 'r' )
plt.hist(quick_report3['Number_of_Casualties'],bins=24,histtype = 'step',color = 'b' )
plt.xlabel('Number of Vehicles and Casualties (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram')
plt.show()
# Crear histogramas (9)
# (0 = transparente; 1 = relleno )
plt.hist(quick_report3['Number_of_Vehicles'],bins=24,histtype = 'stepfilled',color = 'r',alpha=0.5 )
plt.hist(quick_report3['Number_of_Casualties'],bins=24,histtype = 'stepfilled',color = 'b',alpha=0.3 )
plt.xlabel('Number of Vehicles and Casualties (per day)')
plt.ylabel('Frecuency')
plt.title('Vehicles Histogram')
plt.show()
#%%Tablas sobre los datos graficados
# frecuencia y clases 
hist,bins = np.histogram(quick_report3['Number_of_Vehicles'],bins=24)
# probabilidad y clases 
prob,bins = np.histogram(quick_report3['Number_of_Vehicles'],bins=24,normed=True)
# dudas de:  escribir en la terminal de python    ?np.histogram