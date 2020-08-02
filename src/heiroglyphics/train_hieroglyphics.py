from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

from sklearn.model_selection import train_test_split

import random
import gc
import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
# import seaborn as sns
# %matplotlib inline 

class TrainHieroglyphics:

    number_of_classes = None
    model = None
    image = None

    train_imgs = []
    test_imgs = []

    train_dir = ""
    validation_dir = ""

    nrows = 150
    ncolumns = 150
    channels = 3

    def __init__(self):
        # self.number_of_classes = 2
        # self.train_dir = "../data/sample/train"
        # self.validation_dir   = "../data/sample/validation"

        # List data set
        print(os.listdir("../input"))


    def run_all(self):
        self.load_dataset()
        self.load_model()
        self.add_layers_to_model()
        self.train_model()
        # self.store_model()
        print("All done")

    def load_dataset(self):
        train_dir = '../input/train'
        test_dir = '../input/test'

        # train_imgs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir)]  #get full data set
        train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]  #get dog images
        train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  #get cat images

        self.test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images

        self.train_imgs = train_dogs[:2000] + train_cats[:2000]  # slice the dataset and use 2000 in each class
        random.shuffle(self.train_imgs)  # shuffle it randomly

        #Clear list that are useless
        del train_dogs
        del train_cats
        gc.collect()   #collect garbage to save memory
        

    def read_and_process_image(self, list_of_images):
        """
        Returns two arrays: 
            X is an array of resized images
            y is an array of labels
        """
        X = [] # images
        y = [] # labels
        
        for image in list_of_images:
            X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (self.nrows,self.ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
            #get the labels
            if 'dog' in image:
                y.append(1)
            elif 'cat' in image:
                y.append(0)
        
        return X, y


    def load_model(self):
        self.model = ResNet50(include_top=False, weights='imagenet', input_shape=(150,150,3))

        output = self.model.layers[-1].output
        output = Flatten()(output)

        self.model = Model(self.model.input, output)

        for layer in self.model.layers:
            layer.trainable = False
        
        self.model.summary()

    def add_layers_to_model(self):
        input_shape=(150, 150,3)
        model = Sequential()
        model.add(self.model)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=2e-5),
                    metrics=['accuracy'])
        model.summary()

        self.model = model


    def train_model(self):
        print("Train the model using the defined data set")

        X, y = self.read_and_process_image(self.train_imgs)

        del self.train_imgs
        gc.collect()

        X = np.array(X)
        y = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

        del X
        del y
        gc.collect()

        ntrain = len(X_train)
        nval = len(X_val)

        batch_size = 32  

        train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow(X_train, y_train,batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


        history = self.model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=20,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)
   

    # def store_model(self):
    #     print("Store the model")
    #     self.model.save("test.h5")

    # def perform_inference(self):
    #     features = self.model.predict(self.image)
    #     print("Features are: ")
    #     print(features.shape)