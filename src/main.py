from os.path import isdir, join, exists, dirname
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import joblib
import os
from os import listdir
from os.path import isfile, join
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
import numpy as np

from featureExtractor import FeatureExtractor
from imageLoader import batchGenerator


class StartApplication:

    def __init__(self):
        file_dir = dirname(__file__)
        self.dataPath = join(file_dir, "../data/glyphdataset/Dataset")
        self.download_dataset()
        self.intermediatePath = join(file_dir, "../intermediates")
        self.featurePath = join(self.intermediatePath, "features.npy")
        self.labelsPath = join(self.intermediatePath, "labels.npy")
        self.svmPath = join(self.intermediatePath, "svm.pkl")
        self.image_paths = []
        self.labels = []
        self.batch_size = 200
        self.model_training()

    def download_dataset(self):
        """
        Download dataset from 'http://iamai.nl/downloads/GlyphDataset.zip' if zip file does not exist
        """
        if not exists(self.dataPath):
            print("downloading dataset (57.5MB)")
            url = urlopen("http://iamai.nl/downloads/GlyphDataset.zip")
            with ZipFile(BytesIO(url.read())) as z:
                z.extractall(join(self.dataPath, ".."))

    # def convert_dataset(self):
    #     mypath = "test_data/"
    #
    #     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #
    #     print(onlyfiles)
    #
    #     for png in onlyfiles:
    #
    #         print(png)
    #
    #         folder_name = png[7:-4]
    #
    #         # check if folder name exists
    #
    #         if os.path.isdir('clean_data/' + folder_name) == False:
    #             os.makedirs('clean_data/' + folder_name)
    #
    #         # copy image to suitable folder and rename
    #         shutil.move(mypath + "/" + png, 'clean_data/' + folder_name + "/")

    def model_training(self):

        # check if the feature file is present, if so; there is no need to recompute the features
        # The pre-computed features can also be downloaded from http://iamai.nl/downloads/features.npy
        if not isfile(self.featurePath):
            print("indexing images...")
            Steles = [join(self.stelePath, f) for f in listdir(self.stelePath) if isdir(join(self.stelePath, f))]
            for stele in Steles:
                imagePaths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]
                for path in imagePaths:
                    self.image_paths.append(path)
                    self.labels.append(path[(path.rfind("_") + 1): path.rfind(".")])

            featureExtractor = FeatureExtractor()
            features = []
            print("computing features...")
            for idx, (batch_images, _) in enumerate(batchGenerator(self.image_paths, self.labels, self.batch_size)):
                print("{}/{}".format((idx + 1) * self.batch_size, len(self.labels)))
                features_ = featureExtractor.get_features(batch_images)
                features.append(features_)
            features = np.vstack(features)

            labels = np.asarray(self.labels)
            print("saving features...")
            np.save(self.featurePath, features)
            np.save(self.labelsPath, labels)
        else:
            print("loading precomputed features and labels from {} and {}".format(self.featurePath, self.labelsPath))
            features = np.load(self.featurePath)
            labels = np.load(self.labelsPath)

        # on to the SVM trainign phase
        tobeDeleted = np.nonzero(labels == "UNKNOWN")  # Remove the Unknown class from the database
        features = np.delete(features, tobeDeleted, 0)
        labels = np.delete(labels, tobeDeleted, 0)
        numImages = len(labels)
        trainSet, testSet, trainLabels, testLabels = train_test_split(features, labels, test_size=0.20, random_state=42)

        # Training SVM, feel free to use linear SVM (or another classifier for that matter) for faster training, however that will not give the confidence scores that can be used to rank hieroglyphs
        print("training SVM...")
        if 0:  # optinal; either train 1 classifier fast, or search trough the parameter space by training multiple classifiers to sqeeze out that extra 2%
            clf = linear_model.LogisticRegression(C=10000)
        else:
            svr = linear_model.LogisticRegression()
            parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
            clf = GridSearchCV(svr, parameters, n_jobs=8)
        clf.fit(trainSet, trainLabels)

        print(clf)
        print("finished training! saving...")
        joblib.dump(clf, self.svmPath, compress=1)

        prediction = clf.predict(testSet)
        accuracy = np.sum(testLabels == prediction) / float(len(prediction))

        # for idx, pred in enumerate(prediction):
        #     print("%-5s --> %s" % (testLabels[idx], pred))
        print("accuracy = {}%".format(accuracy * 100))

if __name__ == "__main__":
    """
    Start the application
    """
    app = StartApplication();
