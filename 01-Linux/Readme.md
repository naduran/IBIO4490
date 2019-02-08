

Laboratory 01
Natalia Andrea Durán Castro 
201530022
07/02/2019

Report 
1.	The grep command is used to search text, because it process line by line of a text and prints the lines that match a specified pattern. Therefore, it can search lines of text that match with the regular expressions specified.  The syntax of the command is:  grep “string” FILE_PATTERN. In addition, it can be used grep –i to case insensitive search. [1] [2]

2.	The shebangs are command sequences to make a file executable. The meaning of “#!/bin/Python” at the start of the scripts is for interpret how execute a script via Python. [3]

3.	vision@bcv002:~/na.duran$ wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
vision@bcv002:~/na.duran$ tar -xzvf BSR_bsds500.tgz [4]

4.	The disk size of the uncompressed data set is 74128 kb. [5]
vision@bcv002:~/na.duran$ du BSR

Number of imagens (In view of the fact that all images are in format.jpg)
vision@bcv002:~/na.duran/BSR/BSDS500/data/images$ find . -name "*.jpg" | wc –l

Number of imagens: 500 (200 in train, 200 in test and 100 in val).

5.	The resolutions are 321X481 and 481X321

vision@bcv002:~/na.duran/BSR/BSDS500/data/images$ find . -name "*.jpg" -exec identify {} \; | awk '{print $3}' | sort | uniq   

Their format is JPEG. 

vision@bcv002:~/na.duran/BSR/BSDS500/data/images$ find . -name "*.jpg" -exec identify {} \; | awk '{print $2}' | sort | uniq   

6.	In this case, the images with resolution of 481X321 are landscape and the others (321X481) are portrait.  
find . -name "*.jpg" -exec identify {} \; | awk '{print $3}' |grep  "481x321" | wc –l
There are 348 landscape images and 152 portrait images. 

7.	


References
[1] https://www.interserver.net/tips/kb/linux-grep-command-usage-examples/
[2]https://www.thegeekstuff.com/2009/03/15-practical-unix-grep-command-examples/
[3] https://martin-thoma.com/what-does-usrbinpython-mean/
[4] http://ecapy.com/comprimir-y-descomprimir-tgz-tar-gz-y-zip-por-linea-de-comandos-en-linux/index.html
[5] https://www.keopx.net/blog/uso-de-du-para-saber-el-tamano-de-las-carpetas





Sample Exercise: Image database
1.	vision@bcv002:~ mkdir na.duran
2.	vision@bcv002:~$ cp -R data/sipi_images/ na.duran/
3.	http://ecapy.com/comprimir-y-descomprimir-tgz-tar-gz-y-zip-por-linea-de-comandos-en-linux/index.html
a.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf aerials.tar.gz
b.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf misc.tar.gz
c.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf sequences.tar.gz
d.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf textures.tar.gz
4.	159 grayscale images in total.
5.	vision@bcv002:~/na.duran/sipi_images$ nano
# are comments
#!/bin/bash

# go to Home directory
cd ~ # or just cd

cd na.duran
# remove the folder created by a previous run from the script
rm -rf color_images

# create output directory
mkdir color_images

# find all files whose name end in .tif
images=$(find sipi_images -name *.tiff)

#iterate over them
for im in ${images[*]}
do
   # check if the output from identify contains the word "gray"
   identify $im | grep -q -i gray

   # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
   if [ $? -eq 0 ]
   then
      echo $im is gray
   else
      echo $im is color
      cp $im color_images
   fi
done
vision@bcv002:~/na.duran/sipi_images$ chmod u+x find_colors_images.sh
vision@bcv002:~/na.duran/sipi_images$ ./find_colors_images.sh

