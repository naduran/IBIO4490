# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:47:00 2019

@author: NataliaDurán
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
from scipy import ndimage as ndi
try:
    import cv2
except ImportError:
    subprocess.call(['pip','install','opencv-contrib-python'])
import cv2

#Show the image with plt and mpimg
imgo1=mpimg.imread('P2.jpeg')
plt.figure(1)
plt.subplot(3,2,1)
plt.axis("off")
plt.imshow(imgo1)


#Show the image with cv2 and plt
imgo2=cv2.imread('P1.jpeg')
plt.subplot(3,2,2)
plt.axis("off")
plt.imshow(cv2.cvtColor(imgo2, cv2.COLOR_BGR2RGB))


#Apply of high-filter with kernel
img1=cv2.imread('P1.jpeg',0)
data = np.array(img1, dtype=float)
kernel = np.array([[-1, -1, -1],
                   [-1,  30, -1],
                   [-1, -1, -1]])
kernel=1/30*kernel
img1= ndi.convolve(data, kernel)
img1 = img1.astype(np.uint8)
img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
plt.subplot(3,2,3)
plt.axis("off")
plt.imshow(img1)

#Apply of low filter (Gaussian)
img2=cv2.imread('P2.jpeg')
d=270
img2 = cv2.GaussianBlur(img2,(27,27),d)
plt.subplot(3,2,4)
plt.axis("off")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

#Make the hybrid image
img=cv2.addWeighted(img1,0.5,img2,0.5,0)
#Show the image
plt.subplot(3,2,5)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

plt.figure(2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
#Try to see the image more little
#imgp=cv2.pyrDown(img)
#imgp=cv2.pyrDown(imgp)
#imgp=cv2.pyrDown(imgp)
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=80, sharex=True, sharey=True)
#ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#ax[0].imshow(cv2.cvtColor(imgp, cv2.COLOR_BGR2RGB))

#References 
#https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html
#https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
#https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
#https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
#https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
