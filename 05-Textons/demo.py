# -*- coding: utf-8 -*-

#! /usr/bin/python
#It needs to be improve
import sklearn
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix  
import pickle
import matplotlib.pyplot as plt
import os
from cifar10 import load_cifar10
import numpy as np
import sys
sys.path.append('python')
from assignTextons import assignTextons
from fbRun import fbRun
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization
from skimage import color
import random

file = open("textons.pckl",'rb')
textons = pickle.load(file)
file.close()

file = open("model.pckl",'rb')
model = pickle.load(file)
file.close()

test_cf=load_cifar10(meta='cifar-10-batches-py', mode='test')
test_images=test_cf["data"]
test_labels=test_cf["labels"]

N=random.randrange(0,len(test_labels),1)
l=random.sample(range(0,len(test_labels)-1),N)
list_test=range(0,N-1)
from assignTextons import assignTextons

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)
size=np.shape(textons)
k=size[0]
hists_test=np.zeros((len(test_images),k))
tmap_test=np.zeros((len(test_images),32,32))
for d in l
  for a in list_test:    
      tmap_test[a]=assignTextons(fbRun(fb,color.rgb2gray(test_images[d,:,:,:])),textons.transpose())
      hists_test[a]=np.linalg.norm(histc(tmap_test[d].flatten(), np.arange(k))/tmap_test[d].size) #Normalize histograms
    
Prediccion=model.predict(hists_test)
for i in list_test:
    Prediccion[i]=round(Prediccion[i])
Prediccion=Prediccion.astype(int)

c=confusion_matrix(test_labels, Prediccion)
c=c / c.astype(np.float).sum(axis=1) #Normalize matrix

d=0
for i in range(len(c)):
    d=d+c[i,i]
ACA=d/(len(c))
print('ACA test:',ACA)
print(c) 

plt.imshow(c)
plt.colorbar()
toc=time.clock()
plt.show()