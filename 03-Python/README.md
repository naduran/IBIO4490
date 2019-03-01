# Python tutorial
This is a basic tutorial on Python. We aim to cover general topics such as syntax and data structures, packaging, debugging, data loading and visualization.

## Warming up
- [ipython](https://ipython.org/)
- [Jupyter](http://jupyter.org/)

## Additional Info about LaTeX
- [Sharelatex](https://www.sharelatex.com)
- [Overleaf](https://www.overleaf.com/)
- [CVPR Author kit](http://cvpr2019.thecvf.com/files/cvpr2019AuthorKit.tgz)

## FYI - Requierements and dependencies installation
It is recommended to use Python [Anaconda](https://www.continuum.io/downloads) and its powerful package manager, [conda](https://github.com/conda/conda) to install additional packages.

--------------

# Homework

Please select one dataset of your preference. If you do not have one, you might use [BDSD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.htm). Your dataset _SHOULD_ have labels. 

Write a **python** script that does the following things:
1. Download (and untar if the case) your dataset. Here you can be *very* creative *i.e.*, you may use Dropbox, Drive, or anything you can access from a python module. 
2. Choose _randomly_ an specific number (Let's say **N**, N>6) of those images. Resize them to 256x256, and save them in a new folder. 
3. Plot the **N** images with the corresponding labels.
4. Delete the folder previously created. 

Let's explore the last item. On one hand, if you choose a classification problem, you might do it by showing the original images and something that tells the label of each one. For instance:

![fake](imgs/fake.png)

Where the label is depicted at the center of each image.

On the other hand, if you choose a detection/segmentation problem, you might do it by showing the original images with the corresponding groundtruths. For instance:

![bsds](imgs/bsds.png)

Where the first row are the original images, second and third row are the boundaries and segmentation grountruth respectively. 


---

Your script **must** be ONE SINGLE executable script that does all the aforementioned items just by typing: `./run.py`. Every time the script **must** choose different random images. 

Notes:
- Once the dataset has been downloaded, the script must skip step 1. 
- If you make use of a module that is rather uncommon (I do not know, nobody knows, internet is a weird place), just make sure that your script internally install it.
- Print the processing time at the end of the script. _time_ is the module you need for this purpose.

Bonus: 
- I could contemplate the posibility of a bonus if someone does not use the BDSD dataset. Be creative.

# Solution

The scritp is in the folder 03-Python of my repository (run.py)
```
#! /usr/bin/python
import time
tic=time.clock()
import os,random, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
# vision@bcv002:~/na.duran/IBIO4490/03-Python$
try:
    import cv2
except ImportError:
    subprocess.call(['pip','install','opencv-contrib-python'])
    import cv2
directory=os.listdir(os.getcwd())
#Download original dataset[:D (creativity!)]. It has 30 images of horses, cats and owls. The name has the annotations.
#os.system("rm Dataset\ animales-20190214T131444Z-001.zip")
#os.system("rm -rf Dataset\ animales/")
if 'show_im' in directory:
   os.system("rm -rf show_im")
 #Condition to skip step 1
if 'Dataset animales'not in directory: 
#Scritp developed with help of: https://www.kaggle.com/deadskull7/cats-vs-dogs-images
 
    os.system('curl gdrive.sh | bash -s 1wKpuUXbe7OnjxMYyG8r2OFMmpmixaATi')
#https://drive.google.com/file/d/1wKpuUXbe7OnjxMYyG8r2OFMmpmixaATi/view?usp=sharing

    os.system('unzip Dataset\ animales-20190214T131444Z-001.zip')
py=os.getcwd()
show_im=py+ '/'+'show_im'+'/'
    
    #Erase images saved in the last run
for file in os.listdir(py):
  if file.endswith('.jpg'):
    os.remove(file)



    
data_dir=py+'/'+'Dataset animales'+'/'
#List with the path of each image in dataset
data = [data_dir+i for i in os.listdir(data_dir) if 'jpg' in i]
#Number of images to show choosed randomly
N=random.randrange(2,len(data),2)

#List to choose random images of data set
l=random.sample(range(0,len(data)-1),N)
count=0
rand_data=N*[1]
for c in l:
  if count<N:
    rand_data[count]=data[c]
    count=count+1
    print(rand_data[count-1])
    print(data[c])

#Folder to save the random images
os.system('mkdir show_im')

#Copy the random choosed images to the principal folder of python
for filename in rand_data:
    if filename.endswith('.jpg'):
        shutil.copy(filename, py)
        #shutil.copy(filename, show_im)
    print(filename)
    
from PIL import Image, ImageDraw, ImageFont

#Resize and add label to the images choosed
for n in os.listdir(py):
  if n.endswith('.jpg'):
    img = Image.open(n)
    #os.system("rm n")
    img = img.resize((256,256))
    draw = ImageDraw.Draw(img)
    #Problem with font. I had to use fc-list to find 'Latom-Medium.ttf'
    font = ImageFont.truetype('Lato-Medium.ttf', 50)
    
    if 'c' in n:
      m="cat"
    elif 'h' in n:
      m='horse' 
    elif 'b' in n:
      m='owl' 
    
    draw.text((50, 50), m, font=font)
    
    img.save(n,'JPEG')
  
#Move the images to the folder created
for filename in os.listdir(py):
    if filename.endswith('.jpg'):
      #if filename not in os.listdir(data_dir):
        shutil.move(filename, show_im)
        print(filename)
 #Creates a list of the images resized with labels in the folder      
show=[show_im+i for i in os.listdir(show_im) if 'jpg' in i]
#Plot the images together.
plt.figure(1)
for i in range (0,N):
     im=mpimg.imread(show[i])
     plt.subplot(2,N/2,i+1) 
     plt.axis('off')
     plt.imshow(im)
     plt.subplots_adjust(wspace=0, hspace=0)
#Take the time of processing and print it in the terminal
toc=time.clock()
print(toc-tic) #>5.66 s without download the data.
plt.show()
    
#Remove the folder
os.system("rm -rf show_im")



#References 
#[1] https://www.kaggle.com/deadskull7/cats-vs-dogs-images
#[2] https://stackoverflow.com/questions/42535925/python-how-to-dynamically-display-image-in-label-widget-tkinter
#[3] https://stackoverflow.com/questions/24085996/how-i-can-load-a-font-file-with-pil-imagefont-truetype-without-specifying-the-ab
#[4] https://github.com/naduran/IBIO4490/tree/master/03-Python
#[5] https://stackoverflow.com/questions/1465146/how-do-you-determine-a-processing-time-in-python

#Extra work!
#Bad dataset :(
#os.system("rm penguin.tar.gz")
#os.system("rm beaver.tar.gz")
#os.system("rm -rf penguin/")
#os.system("rm -rf beaver/")

#os.system("wget http://www.tamaraberg.com/animalDataset/tarfiles/penguin.tar.gz")
#os.system("wget http://www.tamaraberg.com/animalDataset/tarfiles/beaver.tar.gz")
#import tarfile
#tf = tarfile.open("penguin.tar.gz")
#tf.extractall()

#tf = tarfile.open("beaver.tar.gz")
#tf.extractall()
    

 ```
