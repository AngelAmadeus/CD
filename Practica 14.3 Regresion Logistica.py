#Practica 14: Regresión Logística 3
import numpy as np
import pandas as pd
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
#%%Importación de Datos
data=pd.read_csv('../Data/BD_LogReg_SVM.txt',header=None)
X=data.iloc[:,0:2]
Y=data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
#%%Buscar el polinomio "óptimo"
#Como no se que polinomio me conviene, intento con varios, analizo y luego elijo
ngrado = 15 #Grado del polinomio
grados = np.arange(1,ngrado)
ACCURACY = np.zeros(grados.shape)
PRECISION = np.zeros(grados.shape)
RECALL = np.zeros(grados.shape)
F1 = np.zeros(grados.shape)
NUM_VARIABLES = np.zeros(grados.shape)
#%%Modelo de regresión lineal
for ngrado in grados:
    poly=PolynomialFeatures(ngrado)
    Xasterisco=poly.fit_transform(X) #es el x modificado, el que se le grega la fila de 1's
    logreg = linear_model.LogisticRegression(C=1)
    logreg.fit(Xasterisco,Y) #Entrena el modelo
    Yg=logreg.predict(Xasterisco) #Sacar el "y" estimado
    #Guardar las variables en las matrices
    NUM_VARIABLES[ngrado-1] = len(logreg.coef_[0])
    ACCURACY[ngrado-1] = sk.accuracy_score(Y,Yg) #Emparejamiento Simple
    PRECISION[ngrado-1] = sk.precision_score(Y,Yg) #Precision
    RECALL[ngrado-1] = sk.recall_score(Y,Yg) #Recall
    F1[ngrado-1] = sk.f1_score(Y,Yg) #F1
#%%Visualizar los resultados
plt.plot(grados,ACCURACY)
plt.plot(grados,PRECISION)
plt.plot(grados,RECALL)
plt.plot(grados,F1)
plt.legend(('Accuracy','Precision','Recall','F1'))
plt.grid()
plt.show()
#%%Visualizar el grado de polinomio
plt.bar(grados,NUM_VARIABLES)
plt.title('Relación Grado-Parámetros')
plt.xlabel('Grado del Polinomio')
plt.ylabel('Número de Parámetros (w´s)')
plt.grid()
plt.show()
#(Por lo que se observa en las graficas la respuesta sería el poinomio de grado 4)
#%%Seleccionar el grado óptimo del análisis anterior
ngrado = 4
poly = PolynomialFeatures(ngrado)
Xasterisco = poly.fit_transform(X)
logreg = linear_model.LogisticRegression(C=1)
logreg.fit(Xasterisco,Y)
Yg = logreg.predict(Xasterisco)
sk.accuracy_score(Y,Yg) #Porcentaje de acierto en total, y lo muestra en la terminal
#%% Anlaisiar los coeficientes  más significativos
W = logreg.coef_[0]
plt.bar(np.arange(len(W)),W)
plt.title('Relación Varaible-Valor del Parametro')
plt.xlabel('Número de Varible (x´s)')
plt.ylabel('Valor del Parámetro (w´s)')
plt.show()
#%%Anlaizar los coeficientes más significativos
W = logreg.coef_[0]
Wabs = np.abs(W)
umbral = 0.5 #umbral que indica que tan significante o insignificante es el valor de un parámetro
indx = Wabs>umbral
Xasterisco_seleccionada = Xasterisco[:,indx] #Sub matriz de x asterisco con las variables de los parametros significativos
plt.bar(np.arange(len(W[indx])),W[indx])
plt.title('Relación Varaible-Valor del Parametro Significativos')
plt.xlabel('Número de Varible (x´s)')
plt.ylabel('Valor del Parámetro (w´s)')
plt.show()
#%%Reentrenar el modelo con las variables seleccionadas
logreg_entrenada = linear_model.LogisticRegression(C=1)
logreg_entrenada.fit(Xasterisco_seleccionada,Y)
Yg_entrenado = logreg_entrenada.predict(Xasterisco_seleccionada)
sk.accuracy_score(Y,Yg_entrenado) #Porcentaje de acierto en total, y lo muestra en la terminal
diferencia = sk.accuracy_score(Y,Yg) - sk.accuracy_score(Y,Yg_entrenado)
print('la diferencia en porcentaje de aciertos del modelo entrenado y no entrenado es: ')
print(diferencia)
#Se observa que pese a tener menos variables, el porcentaje de accuracy score entrenado
#y el porcentahe de acierto sin entrenar, es el mismo. Es decir que con menos variables
#se llegó exactamente al mismo resultado. (con umbral de 0.5)
#%% Segundo criterio: Eliminar coeficientes en orden ascendente
indx = np.argsort(Wabs)[::-1] #ordena de forma ascendente
features = np.arange(1,len(indx)) #lsita que indica las variables que seran seleccionadas, es decir que primero hara el modelo con
#una caracteristica (la mas significativa), despues con la primera y la segunda más significativa, y asi sucesivamente
ACCURACY = np.zeros(grados.shape)
PRECISION = np.zeros(grados.shape)
RECALL = np.zeros(grados.shape)
F1 = np.zeros(grados.shape)
for nfeature in features:
    Xasterisco_seleccionada = Xasterisco[:,indx[0:nfeature]]
    logreg = linear_model.LogisticRegression(C=1)
    logreg.fit(Xasterisco_seleccionada,Y)
    Yg=logreg.predict(Xasterisco_seleccionada)
    ACCURACY[nfeature-1] = sk.accuracy_score(Y,Yg) #Emparejamiento Simple
    PRECISION[nfeature-1] = sk.precision_score(Y,Yg) #Precision
    RECALL[nfeature-1] = sk.recall_score(Y,Yg) #Recall
    F1[nfeature-1] = sk.f1_score(Y,Yg) #F1
#%%Visualizar los resultados
plt.plot(features,ACCURACY)
plt.plot(features,PRECISION)
plt.plot(features,RECALL)
plt.plot(features,F1)
plt.title('Resultados Ordenando de Forma Ascendente')
plt.legend(('Accuracy','Precision','Recall','F1'))
plt.xlabel('Número de Features')
plt.ylabel('Porcentaje del indicador')
plt.grid()
plt.show()
#%%Ver la Ecuación
#De acuerdo al segundo criterio, el numero de variables resultantes serían 9
nfeature = 9
names = np.array(poly.get_feature_names()) #Lista de las variables del polinomio
names[indx[0:nfeature]] #lista de las 9 varaibles significativas del modelo
#Bajo el modelo de la regresión logística, y el segundo criterio, se encontro que el polinomio optimo que separa
#los puntos morados de los amarillos, es de grado 4, y que de dicho polinomio de 15
#variables, hay 9 significativas que lo modelan con precision.
#%% Tercer criterio: Eliminar coeficientes en orden descendente
indx = np.argsort(Wabs)[::1] #ordena de forma descendente
features = np.arange(1,len(indx)) #lsita que indica las variables que seran seleccionadas, es decir que primero hara el modelo con
#una caracteristica (la menos significativa), despues con la primera y la segunda menos significativa, y asi sucesivamente
ACCURACY = np.zeros(grados.shape)
PRECISION = np.zeros(grados.shape)
RECALL = np.zeros(grados.shape)
F1 = np.zeros(grados.shape)
for nfeature in features:
    Xasterisco_seleccionada = Xasterisco[:,indx[0:nfeature]]
    logreg = linear_model.LogisticRegression(C=1)
    logreg.fit(Xasterisco_seleccionada,Y)
    Yg=logreg.predict(Xasterisco_seleccionada)
    ACCURACY[nfeature-1] = sk.accuracy_score(Y,Yg) #Emparejamiento Simple
    PRECISION[nfeature-1] = sk.precision_score(Y,Yg) #Precision
    RECALL[nfeature-1] = sk.recall_score(Y,Yg) #Recall
    F1[nfeature-1] = sk.f1_score(Y,Yg) #F1
#%%Visualizar los resultados
plt.plot(features,ACCURACY)
plt.plot(features,PRECISION)
plt.plot(features,RECALL)
plt.plot(features,F1)
plt.legend(('Accuracy','Precision','Recall','F1'))
plt.title('Resultados Ordenando de Forma Descendente')
plt.xlabel('Número de Features')
plt.ylabel('Porcentaje del indicador')
plt.grid()
plt.show()
#%%Ver la Ecuación
#De acuerdo al segundo criterio, el numero de variables resultantes serían 9
nfeature = 14
names = np.array(poly.get_feature_names()) #Lista de las variables del polinomio
names[indx[0:nfeature]] #lista de las 9 varaibles significativas del modelo
#Bajo el modelo de la regresión logística, y el segundo criterio, se encontro que el polinomio optimo que separa
#los puntos morados de los amarillos, es de grado 4, y que de dicho polinomio de 15
#variables, hay 14 significativas que lo modelan con precision.
#%%Visuaizar los datos en una tabla
#ACCURACY = pd.DataFrame(ACCURACY, columns=['ACCURACY'])
#PRECISION = pd.DataFrame(PRECISION, columns=['PRECISION'])
#RECALL = pd.DataFrame(RECALL, columns=['RECALL'])
#F1 =  pd.DataFrame(F1, columns=['F1'])
#NUM_VARIABLES = pd.DataFrame(NUM_VARIABLES, columns=['NUM_VARIABLES'])
#Indicadores_de_Similitud = ACCURACY.join(PRECISION).join(RECALL).join(F1).join(NUM_VARIABLES)
