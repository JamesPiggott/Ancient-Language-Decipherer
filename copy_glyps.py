import os
from os import listdir
from os.path import isfile, join, isdir
import shutil

mypath = "test_copy_data/"

alldirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]

for dirs in alldirs:
    
    if os.path.isdir(mypath + dirs) == True:
        
        allfiles = [f for f in listdir(mypath + dirs) if isfile(join(mypath + dirs, f))]
        
        for glyph in allfiles:
        
            shutil.move(mypath + dirs + "/" + glyph, "../")
        
            print(glyph)
