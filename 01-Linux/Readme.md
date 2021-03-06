


# Introduction to Linux

## Preparation

1. Boot from a usb stick (or live cd), we suggest to use  [Ubuntu gnome](http://ubuntugnome.org/) distribution, or another ubuntu derivative.

2. (Optional) Configure keyboard layout and software repository
   Go to the the *Activities* menu (top left corner, or *start* key):
      -  Go to settings, then keyboard. Set the layout for latin america
      -  Go to software and updates, and select the server for Colombia
3. (Optional) Instead of booting from a live Cd. Create a partition in your pc's hard drive and install the linux distribution of your choice, the installed Os should perform better than the live cd.

## Introduction to Linux

1. Linux Distributions

   Linux is free software, it allows to do all sort of things with it. The main component in linux is the kernel, which is the part of the operating system that interfaces with the hardware. Applications run on top of it. 
   Distributions pack together the kernel with several applications in order to provide a complete operating system. There are hundreds of linux distributions available. In
   this lab we will be using Ubuntu as it is one of the largest, better supported, and user friendly distributions.


2. The graphical interface

   Most linux distributions include a graphical interface. There are several of these available for any taste.
   (http://www.howtogeek.com/163154/linux-users-have-a-choice-8-linux-desktop-environments/).
   Most activities can be accomplished from the interface, but the terminal is where the real power lies.

### Playing around with the file system and the terminal
The file system through the terminal
   Like any other component of the Os, the file system can be accessed from the command line. Here are some basic commands to navigate through the file system

   -  ``ls``: List contents of current directory
   - ``pwd``: Get the path  of current directory
   - ``cd``: Change Directory
   - ``cat``: Print contents of a file (also useful to concatenate files)
   - ``mv``: Move a file
   - ``cp``: Copy a file
   - ``rm``: Remove a file
   - ``touch``: Create a file, or update its timestamp
   - ``echo``: Print something to standard output
   - ``nano``: Handy command line file editor
   - ``find``: Find files and perform actions on it
   - ``which``: Find the location of a binary
   - ``wget``: Download a resource (identified by its url) from internet 

Some special directories are:
   - ``.`` (dot) : The current directory
   -  ``..`` (two dots) : The parent of the current directory
   -  ``/`` (slash): The root of the file system
   -  ``~`` (tilde) :  Home directory
      
Using these commands, take some time to explore the ubuntu filesystem, get to know the location of your user directory, and its default contents. 
   
To get more information about a command call it with the ``--help`` flag, or call ``man <command>`` for a more detailed description of it, for example ``man find`` or just search in google.


## Input/Output Redirections
Programs can work together in the linux environment, we just have to properly 'link' their outputs and their expected inputs. Here are some simple examples:

1. Find the ```passwd```file, and redirect its contents error log to the 'Black Hole'
   >  ``find / -name passwd  2> /dev/null``

   The `` 2>`` operator redirects the error output to ``/dev/null``. This is a special file that acts as a sink, anything sent to it will disappear. Other useful I/O redirection operations are
      -  `` > `` : Redirect standard output to a file
      -  `` | `` : Redirect standard output to standard input of another program
      -  `` 2> ``: Redirect error output to a file
      -  `` < `` : Send contents of a file to standard input
      -  `` 2>&1``: Send error output to the same place as standard output

2. To modify the content display of a file we can use the following command. It sends the content of the file to the ``tr`` command, which can be configured to format columns to tabs.

   ```bash
   cat milonga.txt | tr '\n' ' '
   ```
   
## SSH - Server Connection

1. The ssh command lets us connect to a remote machine identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER (**vision** in our case). The second command allows us to copy files between systems (you will get the actual login information in class).

   ```bash
   
   #connect
   ssh USER@SERVER
   ```

2. The scp command allows us to copy files form a remote server identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER. Following the SERVER information, we add ':' and write the full path of the file we want to copy, finally we add the local path where the file will be copied (remember '.' is the current directory). If we want to copy a directory we add the -r option. for example:

   ```bash
   #copy 
   scp USER@SERVER:~/data/sipi_images .
   
   scp -r USER@SERVER:/data/sipi_images .
   ```
   
   Notice how the first command will fail without the -r option

See [here](ssh.md) for different types of SSH connection with respect to your OS.

## File Ownership and permissions   

   Use ``ls -l`` to see a detailed list of files, this includes permissions and ownership
   Permissions are displayed as 9 letters, for example the following line means that the directory (we know it is a directory because of the first *d*) *images*
   belongs to user *vision* and group *vision*. Its owner can read (r), write (w) and access it (x), users in the group can only read and access the directory, while other users can't do anything. For files the x means execute. 
   ```bash
   drwxr-x--- 2 vision vision 4096 ene 25 18:45 images
   ```
   
   -  ``chmod`` change access permissions of a file (you must have write access)
   -  ``chown`` change the owner of a file
   
## Sample Exercise: Image database

1. Create a folder with your Uniandes username. (If you don't have Linux in your personal computer)

``vision@bcv002:~ mkdir na.duran``

2. Copy *sipi_images* folder to your personal folder. (If you don't have Linux in your personal computer)

``vision@bcv002:~$ cp -R data/sipi_images/ na.duran/``

3.  Decompress the images (use ``tar``, check the man) inside *sipi_images* folder. 

	``http://ecapy.com/comprimir-y-descomprimir-tgz-tar-gz-y-zip-por-linea-de-comandos-en-linux/index.html``
	
	```a.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf aerials.tar.gz```
	
	```b.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf misc.tar.gz```
	
	```c.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf sequences.tar.gz```
	
	```d.	vision@bcv002:~/na.duran/sipi_images$ tar -xzvf textures.tar.gz```


4.  Use  ``imagemagick`` to find all *grayscale* images. We first need to install the *imagemagick* package by typing

    ```bash
    sudo apt-get install imagemagick
    ```
    
    Sudo is a special command that lets us perform the next command as the system administrator
    (super user). In general it is not recommended to work as a super user, it should only be used 
    when it is necessary. This provides additional protection for the system.
    
    ```bash
    find . -name "*.tiff" -exec identify {} \; | grep -i gray | wc -l
    ```
    
    	There are 159 grayscale images in total.
	
    
5.  Create a script to copy all *color* images to a different folder
    Lines that start with # are comments
       
      ```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

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
      
      ```
      -  save it for example as ``find_color_images.sh``
      -  make executable ``chmod u+x`` (This means add Execute permission for the user)
      -  run ``./find_duplicates.sh`` (The dot is necessary to run a program in the current directory)
      
       ``vision@bcv002:~/na.duran/sipi_images$ nano``
        
        ``vision@bcv002:~/na.duran/sipi_images$ chmod u+x find_colors_images.sh``
	
       `` vision@bcv002:~/na.duran/sipi_images$ ./find_colors_images.sh``

## Your turn

1. What is the ``grep``command?


The grep command is used to search text, because it process line by line of a text and prints the lines that match a specified pattern. Therefore, it can search lines of text that match with the regular expressions specified.  The syntax of the command is:  grep “string” FILE_PATTERN. In addition, it can be used grep –i to case insensitive search. [1] [2]



2. What is the meaning of ``#!/bin/python`` at the start of scripts?


The shebangs are command sequences to make a file executable. The meaning of “#!/bin/Python” at the start of the scripts is for interpret how execute a script via Python. [3]



3. Download using ``wget`` the [*bsds500*](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) image segmentation database, and decompress it using ``tar`` (keep it in you hard drive, we will come back over this data in a few weeks).


``vision@bcv002:~/na.duran$ wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz``

``vision@bcv002:~/na.duran$ tar -xzvf BSR_bsds500.tgz [4]``


 
4. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?


The disk size of the uncompressed data set is 74128 kb. [5]

``vision@bcv002:~/na.duran$ du BSR``


Number of imagens (In view of the fact that all images are in format.jpg)

``vision@bcv002:~/na.duran/BSR/BSDS500/data/images$ find . -name "*.jpg" | wc –l``


Number of images: 500 (200 in train, 200 in test and 100 in val).

 
5. What are all the different resolutions? What is their format? Tip: use ``awk``, ``sort``, ``uniq`` 


The resolutions are 321X481 and 481X321


``vision@bcv002:~/na.duran/BSR/BSDS500/data/images$ find . -name "*.jpg" -exec identify {} \; | awk '{print $3}' | sort | uniq  `` 


Their format is JPEG. 


``vision@bcv002:~/na.duran/BSR/BSDS500/data/images$ find . -name "*.jpg" -exec identify {} \; | awk '{print $2}' | sort | uniq ``  



6. How many of them are in *landscape* orientation (opposed to *portrait*)? Tip: use ``awk`` and ``cut``


In this case, the images with resolution of 481X321 are landscape and the others (321X481) are portrait. 

``find . -name "*.jpg" -exec identify {} \; | awk '{print $3}' |grep  "481x321" | wc –l``

There are 348 landscape images and 152 portrait images. 

 
7. Crop all images to make them square (256x256) and save them in a different folder. Tip: do not forget about  [imagemagick](http://www.imagemagick.org/script/index.php).


```#!/bin/bash



# remove the folder created by a previous run from the script
rm -rf cut_images

# create output directory
mkdir cut_images

# copy all files whose name end in .jpg to the new directory
cp -r $(find . -name "*.jpg") / cut_images

# find all files whose name end in .jpg
images=$(find cut_images -name *.jpg)
#iterate over them
for im in ${images[*]}
do
# http://ask.xmodulo.com/crop-image-command-line-linux.html
# image = $(identify $im | awk '{print $1}'
# convert image -crop 256x256+0+0 image
 
 #Crop all the images in a square of 256x256
 convert $im -crop 256x256+0+0 $im 
done
```



# References

[1] https://www.interserver.net/tips/kb/linux-grep-command-usage-examples/


[2]https://www.thegeekstuff.com/2009/03/15-practical-unix-grep-command-examples/


[3] https://martin-thoma.com/what-does-usrbinpython-mean/


[4] http://ecapy.com/comprimir-y-descomprimir-tgz-tar-gz-y-zip-por-linea-de-comandos-en-linux/index.html


[5] https://www.keopx.net/blog/uso-de-du-para-saber-el-tamano-de-las-carpetas


# Report

For every question write a detailed description of all the commands/scripts you used to complete them. DO NOT use a graphical interface to complete any of the tasks. Use screenshots to support your findings if you want to. 

Feel free to search for help on the internet, but ALWAYS report any external source you used.

Notice some of the questions actually require you to connect to the course server, the login instructions and credentials will be provided on the first session. 

## Deadline

We will be delivering every lab through the [github](https://github.com) tool (Silly link isn't it?). According to our schedule we will complete that tutorial on the second week, therefore the deadline for this lab will be specially long **February 7 11:59 pm, (it is the same as the second lab)** 

### More information on

http://www.ee.surrey.ac.uk/Teaching/Unix/ 




