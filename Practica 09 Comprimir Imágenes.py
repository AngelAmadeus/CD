#Práctica 9: Comprimir Imágenes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%% Importar los datos o leer la imagen
img = mpimg.imread('Img_samurai_color.jpg')
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
dim_max = 2 #Reducir las dimensiones
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
