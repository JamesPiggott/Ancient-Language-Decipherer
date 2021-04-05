from os.path import join, exists, dirname
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import os
from os import listdir
from os.path import isfile, join
import shutil


class StartApplication:

    def __init__(self):
        file_dir = dirname(__file__)
        self.dataPath = join(file_dir, "../data/glyphdataset/Dataset")
        while True:
            self.download_dataset()

    def download_dataset(self):
        """
        Download dataset from 'http://iamai.nl/downloads/GlyphDataset.zip' if zip file does not exist
        """
        if not exists(self.dataPath):
            print("downloading dataset (57.5MB)")
            url = urlopen("http://iamai.nl/downloads/GlyphDataset.zip")
            with ZipFile(BytesIO(url.read())) as z:
                z.extractall(join(self.dataPath, ".."))

    def convert_dataset(self):
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
            shutil.move(mypath + "/" + png, 'clean_data/' + folder_name + "/")


if __name__ == "__main__":
    """
    Start the application
    """
    app = StartApplication();