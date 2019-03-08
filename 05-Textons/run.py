# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:10:53 2019

@author: NataliaDurán
"""

#! /usr/bin/python
import matplotlib.pyplot as plt
import os
import time
import pickle
tic=time.clock()
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



k=300

from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,np.hstack((color.rgb2gray(train_images[range(500),:,:,:]))))

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)
list_t=range(0,len(train_labels)-1)

#Calculate texton representation with current texton dictionary

test_cf=load_cifar10(meta='cifar-10-batches-py', mode='test')
test_images=test_cf["data"]
test_labels=test_cf["labels"]
list_test=range(0,len(test_labels)-1)

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
for a in list_test:    
    tmap_test[a]=assignTextons(fbRun(fb,color.rgb2gray(test_images[a,:,:,:])),textons.transpose())
    hists_test[a]=np.linalg.norm(histc(tmap_test[a].flatten(), np.arange(k))/tmap_test[a].size) #Normalize histograms
import sklearn
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing 
from sklearn import metrics


#classifier = KNeighborsClassifier(n_neighbors=k)
## Training model
#h=classifier.fit(hists_train, train_labels )
##Prediction 
#Prediccion = classifier.predict(hists_test)  
#Prediccion_train = classifier.predict(hists_train)  

from sklearn.ensemble import RandomForestClassifier
F = RandomForestClassifier(n_estimators = 100)
F.fit(hists_train,train_labels)
Prediccion=F.predict(hists_test)
Prediccion_train=F.predict(hists_train)

for i in list_test:
    Prediccion[i]=round(Prediccion[i])
for j in list_t:
    Prediccion_train[j]=round(Prediccion_train[j])
Prediccion=Prediccion.astype(int)
Prediccion_train=Prediccion_train.astype(int)

from sklearn.metrics import classification_report, confusion_matrix  
c=confusion_matrix(test_labels, Prediccion)
t=confusion_matrix(train_labels, Prediccion_train)
c=c / c.astype(np.float).sum(axis=1)
t=t / t.astype(np.float).sum(axis=1)

print(classification_report(test_labels, Prediccion))

d=0
l=0
for i in range(len(c)):
    d=d+c[i,i]
    l=l+t[i,i]
ACA=d/(len(c))
ACA_train=l/(len(c))
print('ACA test:',ACA)
print(c) 

toc=time.clock()
print(toc-tic) 
plt.imshow(c)
plt.colorbar()
toc=time.clock()
plt.show()

print('ACA train:',ACA_train)
print(t)
plt.imshow(t)
plt.colorbar()
plt.show()

f=open('hists_train.pckl','wb')
pickle.dump(hists_train, f)
f.close
g=open('train_labels.pckl','wb')
pickle.dump(train_labels, g)
g.close
h=open('textons.pckl','wb')
pickle.dump(textons, h)
h.close
i=open('model.pckl','wb')
pickle.dump(F,i)

#References

#1.https://github.com/affromero/IBIO4490/tree/master/05-Textons