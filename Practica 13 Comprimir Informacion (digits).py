#Práctica 13: Comprimir Informacion (Digits)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#%%Importación de Datos
digits = datasets.load_digits()
#%%Visualizar los Digitos
ndig = 10
for k in np.arange(ndig):
    plt.subplot(2,ndig/2,k+1)
    plt.axis('off')
    plt.imshow(digits.images[k],cmap=plt.cm.gray_r)
    plt.title('Digit %i' % k)
plt.show()
#%% Aplicar el PCA para reducir las dimenciones 
data = digits.data
medias = data.mean(axis=0)
datam = data-medias
datacov = np.cov(datam.transpose())
w, v = np.linalg.eig(datacov)
#%%Seleccionar los componentes principales
dim_max = 59 #Reducir las dimensiones, ver acumulado
indx = np.argsort(w)[::-1]
w = w[indx]
v = v[indx]
porcentaje = w/np.sum(w) #importante ver pocentajes para tomar desicion 
porcentaje_acumulado = np.cumsum(porcentaje)
#Esto no indica que con 59 pixeles se pueda dibujar el digito, en cambio, esto indica que 
#podemos tener la misma información en 59 dimensiones sin perder información, donde las 
#diemnsiones nuevas son combinaciónes de los otros pixeles, en lugar de las 64 dimensiones 
#(pixeles) que habían al inicio.
componentes = w[indx[0:dim_max]]
transformacion = v[:,indx[0:dim_max]]
#%%Proyectarlo a nuevos ejes
#con estos datas trabajan los modelos
data_new = np.matrix(datam)*np.matrix(transformacion)
#%%Seleccionar pixeles o columnas de los datos antes del PCA
#%Analisis de Varianza
var = np.var(data,axis=0)
plt.bar(np.arange(len(var)),var)
plt.show()
#%las variables de acuerdo a mi umbral para reducir los datos
nivel_var = 0
indx1 = var>nivel_var
data_new2 = data[:,indx1]
#%visualizar los nuevos números
img_cero = np.zeros((1,64))
img_cero[0,indx1] = 16 #Colorea de negro lo que si se utiliza
img_cero =np.reshape(img_cero,(8,8))
plt.imshow(img_cero, cmap=plt.cm.gray_r)
plt.show()
#%%Incluir la categoria de los digitos
componentes = w[indx[0:2]] #direcciones de los nuevos ejes o vectores en el espacio
trans = v[:,indx[0:2]] 
data_new3 = np.matrix(datam)*np.matrix(trans)
data_new3 = np.array(data_new3)
#Gráficas
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(data_new3[:,0],data_new3[:,1])
plt.subplot(122)
plt.scatter(data_new3[:,0],data_new3[:,1],c=digits.target)
plt.colorbar()
plt.show()