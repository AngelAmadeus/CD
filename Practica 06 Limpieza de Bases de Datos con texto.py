#Practica 6: Limpieza de Bases de Datos con texto
import numpy as np
import pandas as pd
import string
import datetime
#%% Importación de Datos
data_file = '..\Data\BD_Dirty.csv'
dirty = pd.read_csv(data_file, header = 0)
#%% remover puntuacion
def remove_punctuation(x):
    try: 
        x = ''.join(ch for ch in x if ch not in string.punctuation)
    except:
        pass
    return x
#%%remover digitos(0,1,2,3,4,5,6,7,8,9)
def remove_digits(x):
    try: 
        x = ''.join(ch for ch in x if ch not in string.digits)
    except:
        pass
    return x
#%%Remover letras minusculas
def remove_minletters(x):
    try: 
        x = ''.join(ch for ch in x if ch not in string.ascii_lowercase)
    except:
        pass
    return x
#%%Remover letras mayusculas
def remove_upletters(x):
    try: 
        x = ''.join(ch for ch in x if ch not in string.ascii_uppercase)
    except:
        pass
    return x
#%%Remover espacios en blanco
def remove_whitespaces(x):
    try: 
        x = ''.join(x.split())
    except:
        pass
    return x
#%%Remplazar texto
def replace_text(x,to_replace,replacement):
    try: 
        x = x.replace(to_replace,replacement)    
    except:
        pass
    return x
#%%Convertir a mayusculas
def uppercase_text(x):
    try: 
        x = x.upper()
    except:
        pass
    return x
#%%Convertir a minusculas
def lowercase_text(x):
    try: 
        x = x.lower()
    except:
        pass
    return x
#%%Convertir a formato de fecha
def date_format(x):
    format_year = "%m/%d/%Y"
    try:
        x = datetime.datetime.strptime(x,format_year)
    except:
        pass
    return x
#%%Limpieza 'fechas'
dirty.iloc[:,1] = dirty.iloc[:,1].apply(replace_text,args=('?','-'))
dirty.iloc[:,1] = dirty.iloc[:,1].apply(replace_text,args=('19xx-10-23','MISSING'))
dirty.iloc[:,1] = dirty.iloc[:,1].apply(replace_text,args=('-','/'))
dirty.iloc[:,1] = dirty.iloc[:,1].apply(replace_text,args=('00','19'))
#%%Limpieza de 'Marital'
dirty.iloc[:,2] = dirty.iloc[:,2].apply(uppercase_text)
dirty.iloc[:,2] = dirty.iloc[:,2].fillna(value='MISSING')
#%%Limpieza de 'people'
dirty.iloc[:,3] = dirty.iloc[:,3].apply(remove_digits)
dirty.iloc[:,3] = dirty.iloc[:,3].apply(replace_text,args=('Aa','A'))
dirty.iloc[:,3] = dirty.iloc[:,3].apply(uppercase_text)
dirty.iloc[:,3] = dirty.iloc[:,3].apply(remove_punctuation)
#%%Limpieza de 'ssn'
dirty.iloc[:,4] = dirty.iloc[:,4].apply(remove_punctuation)
dirty.iloc[:,4] = dirty.iloc[:,4].apply(remove_whitespaces)
dirty.iloc[:,4] = dirty.iloc[:,4].apply(remove_minletters)
dirty.iloc[:,4] = dirty.iloc[:,4].fillna(value='MISSING')
for cont in range(len(dirty.index)):
    if len(dirty.iloc[cont,4]) == 9:
        dirty.iloc[cont,4] = dirty.iloc[cont,4]
    else:
        dirty.iloc[cont,4]='MISSING'
#%%Escribir missing en lugar de MISSING
dirty.iloc[:,:] = dirty.iloc[:,:].apply(replace_text,args=('MISSING','missing'))
#%%Convertir una variable de texto a dummy
r = pd.get_dummies(dirty.marital)
#%%Verificar si la edad corresponde con la fecha de nacimiento
#def verify_age(x,index_col_age,index_col_date,index_date):
#    #hoy = datetime.date.today()
#    #actual_year = int(hoy.year)
#    actual_year = 1978
#    format_year = "%m/%d/%Y"
#    dirty.iloc[:,1] = dirty.iloc[:,1].apply(replace_text,args=('missing','01/01/0001'))
#    date_people = datetime.datetime.strptime(dirty.iloc[index_date,1],format_year)
#    birth_year =  int(date_people.year)
#    if actual_year - birth_year == dirty.iloc[index_date,0]:
#        dirty.iloc[index_date,0] = 1
#    else: 
#        dirty.iloc[index_date,0] = 0
#    return dirty.iloc[index_date,0]
#for cont2 in range(len(dirty.index)):
#    verify_age(dirty.iloc[cont2,0],0,1,cont2)
#%%Otra forma de verificar las edades
#hoy = datetime.date.today()
#actual_year = int(hoy.year)
#actual_year = 1976
#dirty.iloc[:,1] = dirty.iloc[:,1].apply(replace_text,args=('missing','01/01/1450'))
#for cont2 in range(len(dirty.index)):
#    format_year = "%m/%d/%Y"
#    date_people = datetime.datetime.strptime(dirty.iloc[cont2,1],format_year)
#    year_people =  int(date_people.year)
#    age_people = dirty.iloc[cont2,0]
#    if actual_year - year_people == age_people:
#        dirty.iloc[cont2,0] = 1
#    else:
#        dirty.iloc[cont2,0] = 0
