# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 07:32:53 2019

@author: NataliaDurán
"""
#import os, random, shutil, urllib, zipfile
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import subprocess
from scipy import ndimage as ndi
from scipy import misc
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    subprocess.call(['pip','install','opencv-contrib-python'])
import cv2
#https://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy


def scaleSpectrum(A):
    return np.real(np.log10(np.absolute(A) + np.ones(A.shape)))

def high(numRows, numCols, sigma, highp=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highp else coefficient

   return np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])
   
def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   #misc.imsave("dft.png", scaleSpectrum(shiftedDFT))

   filteredDFT = shiftedDFT * filterMatrix
   #misc.imsave("filtered-dft.png", scaleSpectrum(filteredDFT))
   return ifft2(ifftshift(filteredDFT))


def highFilter (img, sigma):
      n,m = img.shape
      return filterDFT(img, high(n, m, sigma,highp=True))

def lowFilter (img, sigma):
      n,m = img.shape
      return filterDFT(img, high(n, m, sigma,highp=False))
  
def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
   highPassed = highFilter(highFreqImg, sigmaHigh)
   lowPassed = lowFilter(lowFreqImg, sigmaLow)

   return highPassed + lowPassed


img=mpimg.imread('P2.jpeg')
plt.axis("off")
#plt.imshow(img)
#plt.show()

img=cv2.imread('P1.jpeg')
plt.axis("off")
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()

#img1=cv2.imread('P1.jpeg',0)
#img1=highFilter(img1,9)

#data = np.array(img, dtype=float)
#kernel = np.array([[-1, -1, -1],
#                   [-1,  30, -1],
#                   [-1, -1, -1]])
#kernel=1/30*kernel
#img1= ndi.convolve(data, kernel)
#img1 = img1.astype(np.uint8)
#img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
#plt.imshow(img1)
#plt.show()

#img2=cv2.imread('P2.jpeg',0)
#data = np.array(img, dtype=float)
#kernel = np.array([[0, 0, 0],
#                   [0,  1, 0],
#                   [0, 0, 0]])
#
#kernel=1/9*kernel
#highpass_3x3 = ndi.convolve(data, kernel)
#plt.imshow(highpass_3x3, cmap='gray' )
#plt.show()
#d=270
#img2 = cv2.GaussianBlur(img,(27,27),d)
#plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#plt.show()
img1=ndi.imread('P1.jpeg', flatten=True)
img2=ndi.imread('P2.jpeg', flatten=True)
img=hybridImage(img1,img2,15,15)
misc.imsave("hibrida_filtros.jpeg", np.real(img))
imgh=cv2.imread('hibrida_filtros.jpeg')
plt.axis("off")
plt.imshow(cv2.cvtColor(imgh, cv2.COLOR_BGR2RGB))
plt.show()
#img=cv2.addWeighted(img1,0.5,img2,0.5,0)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()


#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Principal Reference
#https://github.com/j2kun/hybrid-images/blob/master/hybrid-images.py