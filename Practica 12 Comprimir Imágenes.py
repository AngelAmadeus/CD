#Práctica 12: Comprimir Imágenes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%% Importar los datos o leer la imagen
img = mpimg.imread('samurai_color.jpg')
image = plt.imshow(img)
#%% Reordenar la imagen para procesarla con PCA
d = img.shape
img_reshape = np.reshape(img,(d[0]*d[1],d[2]))
#%% Aplicar el PCA para reducir las dimensiones 
data = img_reshape
medias = data.mean(axis=0)
datam = data-medias
datacov = np.cov(datam.transpose())
w, v = np.linalg.eig(datacov)
#%%Seleccionar los componentes principales
dim_max = 3 #Reducir las dimensiones
indx = np.argsort(w)[::-1]
w = w[indx]
v = v[indx]
porcentaje = w/np.sum(w) #importante ver pocentajes para tomar desicion 
porcentaje_acumulado = np.cumsum(porcentaje)
componentes = w[indx[0:dim_max]]
transformacion = v[:,indx[0:dim_max]]
#%%Proyectarlo a nuevos ejes
#con estos datas trabajan los modelos
data_new = np.matrix(datam)*np.matrix(transformacion)
#%%Recuperar los datos
data_r = np.matrix(data_new)*np.matrix(transformacion.transpose())+medias
#%%Visualizar la imagen
#Las imagenes png tienen las 4 matrices RGBK y las jpg solo RGB
img_r = img.copy()
img_r[:,:,0] = data_r[:,0].reshape((d[0]),(d[1])) #R
img_r[:,:,1] = data_r[:,1].reshape((d[0]),(d[1])) #G
img_r[:,:,2] = data_r[:,2].reshape((d[0]),(d[1])) #B
#img_r[:,:,3] = data_r[:,3].reshape((d[0]),(d[1])) #K
img_r[img_r<0]=0
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_r)
plt.show()
###############################################################################
#%% Importar los datos o leer la imagen
img2 = mpimg.imread('samurai_negro.jpg')
image2 = plt.imshow(img2)
#%% Reordenar la imagen para procesarla con PCA
d2 = img2.shape
img_reshape2 = np.reshape(img2,(d2[0]*d2[1],d2[2]))
#%% Aplicar el PCA para reducir las dimensiones 
data2 = img_reshape2
medias2 = data2.mean(axis=0)
datam2 = data2-medias2
datacov2 = np.cov(datam2.transpose())
w2, v2 = np.linalg.eig(datacov2)
#%%Seleccionar los componentes principales
dim_max2 = 3 #Reducir las dimensiones
indx2 = np.argsort(w2)[::-1]
w2 = w2[indx2]
v2 = v2[indx2]
porcentaje2 = w2/np.sum(w2) #importante ver pocentajes para tomar desicion 
porcentaje_acumulado2 = np.cumsum(porcentaje2)
componentes2 = w2[indx2[0:dim_max2]]
transformacion2 = v2[:,indx2[0:dim_max2]]
#%%Proyectarlo a nuevos ejes
#con estos datos trabajan los modelos
data_new2 = np.matrix(datam2)*np.matrix(transformacion2)
#%%A
A = np.matrix(data_new)*np.matrix(transformacion.transpose())
#%%Recuperar los datos
data_r2 = np.matrix(datam2)*np.matrix(transformacion.transpose())+medias
#%%Visualizar la imagen
#Las imagenes png tienen las 4 atrices RGBK y las jpg solo RGB
img_r2 = img2.copy()
img_r2[:,:,0] = data_r2[:,0].reshape((d2[0]),(d2[1])) #R
img_r2[:,:,1] = data_r2[:,1].reshape((d2[0]),(d2[1])) #G
img_r2[:,:,2] = data_r2[:,2].reshape((d2[0]),(d2[1])) #B
#img_r2[:,:,3] = data_r2[:,3].reshape((d2[0]),(d2[1])) #K
img_r2[img_r2<0]=0
plt.subplot(121)
plt.imshow(img2)
plt.subplot(122)
plt.imshow(img_r2)
plt.show()