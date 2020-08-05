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

    train_generator = None
    val_generator = None

    train_dir = ""


    def __init__(self):
        print(os.listdir("input/"))
        self.run_all()


    def run_all(self):
        self.load_dataset()
        self.load_model()
        self.add_layers_to_model()
        self.train_model()
        # self.store_model()


    def load_dataset(self):
        self.train_dir = '../data/train_data'

        datagen = ImageDataGenerator(rescale=1./255,   
                                            rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

        self.train_generator = datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150), 
            subset='training'
        )

        self.val_generator = datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            subset='validation'
        )
        


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
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=2e-5),
                    metrics=['accuracy'])
        model.summary()

        self.model = model


    def train_model(self):
        print("Train the model using the defined data set")

        history = self.model.fit_generator(self.train_generator,
                            #   steps_per_epoch=32,
                              epochs=20,
                              validation_data=self.val_generator
                            #   validation_steps=32
                              )

   

    # def store_model(self):
    #     print("Store the model")
    #     self.model.save("test.h5")

    # def perform_inference(self):
    #     features = self.model.predict(self.image)
    #     print("Features are: ")
    #     print(features.shape)