# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:10:21 2019

@author: annie
"""
#directory=os.listdir(os.getcwd())
#if 'cifar-10-batches-py'not in directory:
  #l=1   
  #print(l)  
  #os.system('wget http://157.253.196.67/BSDS_small.zip')
  #os.system('7z x BSDS_small.zip')
 
def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    import matplotlib.pyplot as plt
    import os
    import sklearn
    from sklearn import preprocessing 
    from sklearn import metrics
    from PIL import Image  
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    k=numberOfClusters
    rgbImage = plt.imread(rgbImage)
    if colorSpace == "hsv":
    # Si la imagen quiere pasarse a HSV 
        rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2HSV)
# Si la imagen quiere pasarse a LAB   
    elif colorSpace == "lab":
        rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2LAB)
    elif colorSpace == "rgb+xy"
        rgbImage = (cv2.cvtColor(rgbImage, cv2.COLOR_RGB2XYZ)+rgbImage)
    elif colorSpace == "lab+xy"
        rgbImage = (cv2.cvtColor(rgbImage, cv2.COLOR_RGB2XYZ) + cv2.cvtColor(rgbImage, cv2.COLOR_RGB2LAB))
    elif colorSpace == "hsv+xy"
        rgbImage = (rgbImage, cv2.COLOR_RGB2HSV + rgbImage, cv2.COLOR_RGB2XYZ)
    else:
        RGB = rgbImage 
#http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html   
    if  "watersheds" == clusteringMethod:
        from skimage.color import rgb2gray
        from skimage.filters import sobel
        from skimage.segmentation import  watershed
        gradient = sobel(rgb2gray(rgbImage))
        segmentation =  watershed(gradient, markers=350, compactness=0.001)
#https://www.geeksforgeeks.org/gaussian-mixture-model/ 
    elif "gaussian" == clusteringMethod):
        import sklearn
        from  sklearn  import  mixture
        gmm =  sklearn.mixture.GaussianMixture(n_components=k, init_params='random')
        gmm.fit((rgbImage.reshape((rgbImage.size, 1))))
        dim = rgbImage.shape
        labels1 = gmm.predict((rgbImage.reshape((rgbImage.size, 1))))  
        segmentation = labels1.reshape(dim[0],dim[1],3) 
#https://www.geeksforgeeks.org/gaussian-mixture-model/   
    elif "knn" == clusteringMethod):
        import sklearn
        from  sklearn  import  mixture
        gmm =  sklearn.mixture.GaussianMixture(n_components=k, init_params='kmeans')
        gmm.fit((rgbImage.reshape((rgbImage.size, 1))))

        dim = rgbImage.shape
        labels1 = gmm.predict((rgbImage.reshape((rgbImage.size, 1))))  
        segmentation = labels1.reshape(dim[0],dim[1],3)    
else:
    segmentation = cv2.Canny(rgbImage, 25, 200)
return segmentation

