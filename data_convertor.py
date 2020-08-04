import os
from os import listdir
from os.path import isfile, join
import shutil


mypath = "test_data/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)

for png in onlyfiles:

    print(png)

    folder_name = png[7:-4]

    # check if folder name exists

    if os.path.isdir('clean_data/' + folder_name) == False:
        os.makedirs('clean_data/' + folder_name)


    # copy image to suitable folder and rename
    shutil.move(mypath +"/" +png,'clean_data/' + folder_name+"/")  