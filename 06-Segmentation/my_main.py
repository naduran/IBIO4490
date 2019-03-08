# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:44:05 2019

@author: annie
"""
directory=os.listdir(os.getcwd())
if 'cifar-10-batches-py'not in directory:
 l=1   
  print(l)  
  os.system('wget http://157.253.196.67/BSDS_small.zip')
  os.system('7z x BSDS_small.zip')
segmentation1=segmentByClustering( gato, rgb, watersheds, 50) 
segmentation1=segmentByClustering( gato, rgb, watersheds, 50)  
segmentationc=segmentByClustering( gato, rgb, alo, 50)   
  
  plt.figure
plt.subplot(1,3,1)
plt.imshow(rgbImage)
plt.axis('off')
plt.title('Original Im')
plt.subplot(1,3,2)
plt.imshow(segmentation1)
plt.axis('off')
plt.title('Segmentation with 50')
plt.subplot(1,3,3)
plt.imshow(segmentation)
plt.axis('off')
plt.title('Segmentation with 350')

plt.imsave('nova.jpg', nova )

plt.figure
plt.subplot(1,3,1)
plt.imshow(rgbImage)
plt.axis('off')
plt.title('Original Im')
plt.subplot(1,3,2)
plt.imshow(segmentationc)
plt.axis('off')
plt.title('Segmentation with Canny')
