# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:10:53 2019

@author: NataliaDurán
"""

#! /usr/bin/python
import matplotlib.pyplot as plt
import os
#import ipdb
directory=os.listdir(os.getcwd())
if 'cifar-10-batches-py'not in directory:
  l=1   
  print(l)  
  url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  os.system('wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
  os.system('tar -xzvf cifar-10-python.tar.gz')
#print(directory)
#ipdb.set_trace()
train_cf=[]
from cifar10 import load_cifar10

#Load all the data
train_cf=load_cifar10(meta='cifar-10-batches-py', mode=5)

train_images=train_cf["data"]
train_labels=train_cf["labels"]

import sys
sys.path.append('python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

#Load sample images from disk
from skimage import color



k=15*9

from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,np.hstack((color.rgb2gray(train_images[range(500),:,:,:]))))

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)
list_t=range(len(train_labels))
#Calculate texton representation with current texton dictionary

test_cf=load_cifar10(meta='cifar-10-batches-py', mode='test')
test_images=test_cf["data"]
test_labels=test_cf["labels"]


from assignTextons import assignTextons

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)
hists_train=np.zeros((len(train_images),k))
tmap_train=np.zeros((len(train_images),32,32))

hists_test=np.zeros((len(test_images),k))
tmap_test=np.zeros((len(test_images),32,32))
for i in list_t:
    tmap_train[i]=assignTextons(fbRun(fb,color.rgb2gray(train_images[i,:,:,:])),textons.transpose())
    hists_train[i]=np.linalg.norm(histc(tmap_train[i].flatten(), np.arange(k))/tmap_train[i].size) #Normalize histograms
    
    tmap_test[i]=assignTextons(fbRun(fb,color.rgb2gray(test_images[i,:,:,:])),textons.transpose())
    hists_test[i]=np.linalg.norm(histc(tmap_test[i].flatten(), np.arange(k))/tmap_test[i].size) #Normalize histograms
import sklearn
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing 
from sklearn import metrics
#k = 15*9
classifier = KNeighborsClassifier(n_neighbors=k)
# Entrenar el modelo
h=classifier.fit(hists_train, train_labels )
#Predicciones   
Prediccion = classifier.predict(hists_test)  
# Evaluar el algoritmo 
from sklearn.metrics import classification_report, confusion_matrix  
c=confusion_matrix(test_labels, Prediccion)
c=c / c.astype(np.float).sum(axis=1)
print(c) 
print(classification_report(train_labels, Prediccion))

