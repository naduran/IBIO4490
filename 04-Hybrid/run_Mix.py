# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
punto py 
"""
#foto1=Image.open("C:\\Users\\annie\\Documents\\Lab_imagenes_4.jpg")
#Mostrar la imagen 
#foto1.show()
#foto_cop = cv2.resize(foto3,(780,780))
from PIL import Image
import numpy as np 
import cv2
import matplotlib.pyplot as plt


foto3 = cv2.imread("Lab_imagenes_4.jpg")
foto3 = cv2.resize(foto3, (512,512))


# Se hace la piramide gausiana y seguido la piramide laplaciana para de 
#esta manera se puedan guardar los detalles    
pyrone = cv2.pyrDown(foto3)
laplaone = cv2.subtract(foto3 ,cv2.pyrUp(pyrone))
   
pyrtwo = cv2.pyrDown(pyrone)
laplatwo = cv2.subtract(pyrone ,cv2.pyrUp(pyrtwo))
    
   
pyrthree = cv2.pyrDown(pyrtwo)
laplathree = cv2.subtract(pyrtwo ,cv2.pyrUp(pyrthree))
     
    
pyrfour = cv2.pyrDown(pyrthree)
laplafour = cv2.subtract(pyrthree ,cv2.pyrUp(pyrfour))
    
          
pyrfive = cv2.pyrDown(pyrfour)
laplafive = cv2.subtract(pyrfour , cv2.pyrUp(pyrfive))
    
 
    
pyrsix = cv2.pyrDown(pyrfive)
laplasix = cv2.subtract(pyrfive ,cv2.pyrUp(pyrsix))
    
    
   
#   se suben los laplacianos para obtener la imagen original
upone = cv2.pyrUp(laplasix)
    
uptwo =cv2.add( laplafive, upone)
uptwo = cv2. pyrUp(uptwo)
    
upthree = cv2.add(uptwo ,laplafour )
upthree = cv2. pyrUp(upthree)
    
upfour = cv2.add(upthree ,laplathree )
upfour = cv2. pyrUp(upfour)
 
upfive = cv2.add(upfour ,laplatwo )
upfive = cv2. pyrUp(upfive)
    
upsix = cv2.add(upfive ,laplaone)
upsix = cv2. pyrUp(upsix)
    
cv2.imwrite('up6.jpg',upsix)
up6s = Image.open('up6.jpg')
plt.imshow(up6s)
    
#Para mostrar las imagenes

cv2.imwrite('pyrone.jpg',pyrone)
cv2.imwrite('laplaone.jpg',laplaone)
pyrones = Image.open('pyrone.jpg')
plt.imshow(pyrones)
laplaones = Image.open('laplaone.jpg')
plt.imshow(laplaones)
    
cv2.imwrite('pyr2.jpg',pyrtwo)
cv2.imwrite('lapla2.jpg',laplatwo)
pyr2s = Image.open('pyr2.jpg')
plt.imshow(pyr2s)
lapla2s = Image.open('lapla2.jpg')
plt.imshow(lapla2s)
    
cv2.imwrite('pyr3.jpg',pyrthree)
cv2.imwrite('lapla3.jpg',laplathree)
pyr3s = Image.open('pyr3.jpg')
plt.imshow(pyr3s)
laplaones = Image.open('lapla3.jpg')
plt.imshow(lapla3s)
    
cv2.imwrite('pyr4.jpg',pyrone)
cv2.imwrite('lapla4.jpg',laplaone)
pyr4s = Image.open('pyr4.jpg')
plt.imshow(pyr4s)
lapla4s = Image.open('lapla4.jpg')
plt.imshow(lapla4s)
     
cv2.imwrite('pyr5.jpg',pyrone)
cv2.imwrite('lapla5.jpg',laplaone)
pyr5s = Image.open('pyr5.jpg')
plt.imshow(pyr5s)
lapla5s = Image.open('lapla5.jpg')
plt.imshow(lapla5s)
    
cv2.imwrite('pyr6.jpg',pyrone)
cv2.imwrite('lapla6.jpg',laplaone)
pyro6s = Image.open('pyr6.jpg')
plt.imshow(pyr6s)
laplaones = Image.open('lapla6.jpg')
plt.imshow(lapla6s)
    
#subir la piramide
cv2.imwrite('up1.jpg',upone)
up1s = Image.open('up1.jpg')
plt.imshow(up1s)
    
cv2.imwrite('up2.jpg',uptwo)
up2s = Image.open('up2.jpg')
plt.imshow(up2s)
    
cv2.imwrite('up3.jpg',upthree)
up3s = Image.open('up3.jpg')
plt.imshow(up3s)
    
       
cv2.imwrite('up4.jpg',upfour)
up4s = Image.open('up4.jpg')
plt.imshow(up1s)
    
cv2.imwrite('up5.jpg',upfive)
up5s = Image.open('up5.jpg')
plt.imshow(up5s)
    

    
