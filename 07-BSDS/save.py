#!/home/afromero/anaconda3/bin/ipython
#! /usr/bin/python

import os
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

directory=os.listdir(os.getcwd())
if 'BSR' not in directory:
  os.system('wget http://bcv001.uniandes.edu.co/BSDS500FastBench.tar.gz')
  os.system('tar -xzvf BSDS500FastBench.tar.gz BSR/')

def SegmentbyClustering(rgbImage, clusteringMethod,k):
  if  "watersheds" == clusteringMethod:
        from skimage.color import rgb2gray
        from skimage.filters import sobel
        from skimage.segmentation import  watershed
        gradient = sobel(rgb2gray(rgbImage))
        segmentation =  watershed(gradient, markers=k, compactness=0.001)
#https://www.geeksforgeeks.org/gaussian-mixture-model/ 
  elif "gaussian" == clusteringMethod:
        import sklearn
        from  sklearn  import  mixture
        gmm =  sklearn.mixture.GaussianMixture(n_components=k, init_params='random')
        gmm.fit((rgbImage.reshape((rgbImage.size, 1))))
        dim = rgbImage.shape
        labels1 = gmm.predict((rgbImage.reshape((rgbImage.size, 1))))  
        segmentation = labels1.reshape(dim[0],dim[1],3) 
#https://www.geeksforgeeks.org/gaussian-mixture-model/   
  elif "knn" == clusteringMethod:
        import sklearn
        from  sklearn  import  mixture
        gmm =  sklearn.mixture.GaussianMixture(n_components=k, init_params='kmeans')
        gmm.fit((rgbImage.reshape((rgbImage.size, 1))))

        dim = rgbImage.shape
        labels1 = gmm.predict((rgbImage.reshape((rgbImage.size, 1))))  
        segmentation = labels1.reshape(dim[0],dim[1],3)    
  else:
    segmentation = rgbImage
  return segmentation.astype(np.uint8)

os.chdir('BSR/BSDS500/data/images/test/')
images=os.listdir()

k=90
for i in range(len(images)-2):
  img=plt.imread(images[i],'JPEG')

  ticw=time.time()
  g1=SegmentbyClustering(img,'gaussian',k-30)
  tocw=time.time()
  print(images[i])
  print(tocw-ticw)

  tick=time.time()
  g2=SegmentbyClustering(img,'gaussian',k)
  tock=time.time()
  print(tock-tick)

  ticg=time.time()
  g=SegmentbyClustering(img,'gaussian',k+150)
  tocg=time.time()
  print(tocg-ticg)

#plt.figure()
#plt.subplot(4,1,1)
#plt.imshow(img)
#plt.subplot(4,1,2)
#plt.imshow(g1, cmap=plt.get_cmap('tab20b'))
#plt.subplot(4,1,3)
#plt.imshow(g2, cmap=plt.get_cmap('tab20b'))
#plt.subplot(4,1,4)
#plt.imshow(g, cmap=plt.get_cmap('tab20b'))
#plt.show()


  os.chdir('../../../../..')
  segm = np.zeros((3,), dtype=np.object)
  ori_dir=os.getcwd()
  dir_comp=os.listdir(ori_dir)
  if 'segm-test' not in dir_comp:
    os.mkdir('segm-test')
  os.chdir('segm-test')

  segm[0] = g1  
  segm[1] = g2
  segm[2] = g
  name=images[i]+'.mat'
  sio.savemat(name, {'segm':segm})
  os.chdir('..')
  os.chdir('BSR/BSDS500/data/images/test/')
#References

#https://docs.scipy.org/doc/scipy/reference/tutorial/io.html
