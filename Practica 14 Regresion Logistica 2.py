#Practica 14: Regresión Logística 2
import numpy as np
import pandas as pd
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
#%%Importación de Datos
data=pd.read_csv('../Data/ex2data2.txt',header=None)
X=data.iloc[:,0:2]
Y=data.iloc[:,2]
plt.scatter(X[0],X[1],c=Y)
plt.title('Cúmulo de Datos')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
#%% Inicializar variables
ngrado_1=13 #Grado del polinomio para la forma 1
grados_1 = np.arange(1,ngrado_1)
ngrado_2=13 #Grado del polinomio para la forma 2
grados_2 = np.arange(1,ngrado_2)
poly_1=PolynomialFeatures(ngrado_1)
poly_2=PolynomialFeatures(ngrado_2)
Xasterisco_1=poly_1.fit_transform(X) #es el x modificado, el que se le agrega la fila de 1's
Xasterisco_2=poly_2.fit_transform(X) #es el x modificado, el que se le agrega la fila de 1's
#%%Formas de Modelar la regresión Logística
#Para la forma 1
logreg_1=linear_model.LogisticRegression(C=1e20) #Forma 1 (polinomios de grado pequeño)
logreg_1.fit(Xasterisco_1,Y)
Yg_1=logreg_1.predict(Xasterisco_1) # Y estimada (Y gorrito)
#Para la forma 2
logreg_2=linear_model.LogisticRegression(C=1) #Forma 2 (polinomios de grado grande)
#La forma 2 es preferible, evita overfitting y evitas errores en todo el proceso
logreg_2.fit(Xasterisco_2,Y)
Yg_2=logreg_2.predict(Xasterisco_2) # Y estimada (Y gorrito)
#%%Interpretacion de resultados
#lo que se genera es una supercicie, la cual se corta en el eje z = 0 dando resultado
#a las líneas que encierran o limitan los grupos. Aqui se crean los valores que tomará
#la superficie para despues ajustarse a los modelos.
x1=np.arange(-1,1,0.01)
x2=np.arange(-1,1,0.01)
#Todos los puntos posibles 
X1,X2=np.meshgrid(x1,x2)
m,n=np.shape(X1)
X1r=np.reshape(X1,(m*n,1))
m,n=np.shape(X2)
X2r=np.reshape(X2,(m*n,1))
#%%Nueva tabla
Xnew=np.append(X1r,X2r, axis=1) #axis=1 las pega por filas, hace dos columnas
Xasterisco_new_1=poly_1.fit_transform(Xnew)
Xasterisco_new_2=poly_2.fit_transform(Xnew)
Yg_1=logreg_1.predict(Xasterisco_new_1)
Yg_2=logreg_2.predict(Xasterisco_new_2)
#%% Acomodarlo a su forma original
#Graficar la magnitud
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,3))
#Forma 1
Z_1=np.reshape(Yg_1,(m,n))
Z_1=np.round(Z_1)
axes[0].contour(X1,X2,Z_1)
axes[0].scatter(X[0],X[1],c=Y)
axes[0].set_title('Fronteras entre x1 y x2 (Forma 1)')
#Forma 2
Z_2=np.reshape(Yg_2,(m,n))
Z_2=np.round(Z_2)
axes[1].contour(X1,X2,Z_2)
axes[1].scatter(X[0],X[1],c=Y)
axes[1].set_title('Fronteras entre x1 y x2 (Forma 2)')
#Formato
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
plt.show()
#%% Medir el desempeño del modelo
#Para la forma 1
Yg_1 = logreg_1.predict(Xasterisco_1) #prediccion del modelo
cfm_1 = sk.confusion_matrix(Y,Yg_1) #matriz de confusión
accuracy_1 = sk.accuracy_score(Y,Yg_1)
precision_1 = sk.precision_score(Y,Yg_1)
recall_1 = sk.recall_score(Y,Yg_1)
f1_1 = sk.f1_score(Y,Yg_1)
num_variables_1 = np.zeros(grados_1.shape)#cantidad de variables del polinomio
#Para la forma 2
Yg_2 = logreg_2.predict(Xasterisco_2) #prediccion del modelo
cfm_2 = sk.confusion_matrix(Y,Yg_2) #matriz de confusión
accuracy_2 = sk.accuracy_score(Y,Yg_2)
precision_2 = sk.precision_score(Y,Yg_2)
recall_2 = sk.recall_score(Y,Yg_2)
f1_2 = sk.f1_score(Y,Yg_2)
num_variables_2 = np.zeros(grados_2.shape) #cantidad de variables del polinomio
#Nota: Los valores de la matriz de confusión representan lo siguiente
    #(primer componente es y_estimada (filas), y segundo componente y_real (columnas))
    #(0,0) = TrueNegative
    #(0,1) = FalseNegative
    #(1,0) = FalsePositive
    #(1,1) = TruePositive
#Sabiendo lo anterior, hay 4 indicadores que nos ayudan a medir el desempeño del modelo
    #Accuracy = (TP+TN)/(TP + FP + FN + TN), esto es igual al Emparejamiento Simple
    #Precision = (TP)/(TP + FP)
    #Recall = (TP)/(TP + FN)
    #F1 = (2*Precision*Recall)/(Precision + Recall)
#Graficar la magnitud
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,3))
#Forma 1
W_1 = logreg_1.coef_
axes[0].bar(np.arange(len(W_1[0])),W_1[0])
axes[0].set_title('Caracteristicas del Modelo (Forma 1)')
#Forma 2
W_2 = logreg_2.coef_
axes[1].bar(np.arange(len(W_2[0])),W_2[0])
axes[1].set_title('Caracteristicas del Modelo (Forma 2)')
#Formato
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xlabel('Número de Varible (x´s)')
    ax.set_ylabel('Valor del Coeficiente (w´s)')
plt.show()
#%%Buscar el polinomio "optimo". Esto es para la forma 2 es decir, donde C = 1
#Como no se que polinomio me conviene, intento con varios, analizo y luego elijo.
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
#%% Anlaizar los coeficientes más significativos y reducirlo
#Graficar Valores
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,3))
# Visualizar los resultados
axes[0].plot(grados,ACCURACY)
axes[0].plot(grados,PRECISION)
axes[0].plot(grados,RECALL)
axes[0].plot(grados,F1)
axes[0].legend(('Accuracy','Precision','Recall','F1'))
axes[0].set_title('Resultados de los Indicadores')
axes[0].set_ylabel('Porcentaje del Índice')
#Visualizar el grado de polinomio
W_sig = logreg.coef_[0]
Wabs = np.abs(W_sig)
umbral = 0.5 #umbral que indica que tan significante o insignificante es el valor de un parámetro
indx = Wabs>umbral
Xasterisco_seleccionada = Xasterisco[:,indx] #Sub matriz de x asterisco con las variables de los parametros significativos
axes[1].plot(grados,NUM_VARIABLES)
axes[1].set_title('Grado del Polinomio')
axes[0].set_ylabel('Número de Parámetros (w´s)')
#(Por lo que se observa en las graficas la respuesta sería el poinomio de grado 2 ó 4 ó 6)
#Formato
for ax in axes:
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_xlabel('Grado del Polinomio')
    ax.set_ylabel('Valor del Parámetro (w´s)')
plt.show()
#%%Seleccionar el grado óptimo del análisis anterior
ngrado = 4
poly = PolynomialFeatures(ngrado)
Xasterisco = poly.fit_transform(X)
logreg = linear_model.LogisticRegression(C=1)
logreg.fit(Xasterisco,Y)
Yg = logreg.predict(Xasterisco)
sk.accuracy_score(Y,Yg) #Porcentaje de acierto en total, y lo muestra en la terminal
#%% Anlaizar los coeficientes más significativos y reducirlo
#Graficar Valores
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,3))
# Anlaizar los coeficientes
W = logreg.coef_[0]
axes[0].bar(np.arange(len(W)),W)
axes[0].set_title('Relación Varaible-Valor del Parametro')
#Reducido
W_sig = logreg.coef_[0]
Wabs = np.abs(W_sig)
umbral = 0.5 #umbral que indica que tan significante o insignificante es el valor de un parámetro
indx = Wabs>umbral
Xasterisco_seleccionada = Xasterisco[:,indx] #Sub matriz de x asterisco con las variables de los parametros significativos
axes[1].bar(np.arange(len(W[indx])),W[indx])
axes[1].set_title('Relación Varaible-Valor del Parametro Significativos')
#Formato
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xlabel('Número de Varible (x´s)')
    ax.set_ylabel('Valor del Parámetro (w´s)')
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
#una caracteristica (la mpas significativa), despuesde con la primera y la segunda más significativa, y asi sucesivamente
ACCURACY = np.zeros(grados.shape)
PRECISION = np.zeros(grados.shape)
RECALL = np.zeros(grados.shape)
F1 = np.zeros(grados.shape)
for nfeatures in features:
    Xasterisco_seleccionada = Xasterisco[:,indx[0:nfeatures]]
    logreg = linear_model.LogisticRegression(C=1)
    logreg.fit(Xasterisco_seleccionada,Y)
    Yg=logreg.predict(Xasterisco_seleccionada)
    ACCURACY[nfeatures-1] = sk.accuracy_score(Y,Yg) #Emparejamiento Simple
    PRECISION[nfeatures-1] = sk.precision_score(Y,Yg) #Precision
    RECALL[nfeatures-1] = sk.recall_score(Y,Yg) #Recall
    F1[nfeatures-1] = sk.f1_score(Y,Yg) #F1
#%%Visuaizar los datos en una tabla
#ACCURACY = pd.DataFrame(ACCURACY, columns=['ACCURACY'])
#PRECISION = pd.DataFrame(PRECISION, columns=['PRECISION'])
#RECALL = pd.DataFrame(RECALL, columns=['RECALL'])
#F1 =  pd.DataFrame(F1, columns=['F1'])
#NUM_VARIABLES = pd.DataFrame(NUM_VARIABLES, columns=['NUM_VARIABLES'])
#Indicadores_de_Similitud = ACCURACY.join(PRECISION).join(RECALL).join(F1).join(NUM_VARIABLES)