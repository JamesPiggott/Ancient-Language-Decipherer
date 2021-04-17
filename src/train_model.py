# Source project @ https://www.tensorflow.org/tutorials/load_data/images

import os
from os.path import isdir, isfile, join, exists, dirname
from os import listdir
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import tensorflow as tf
from tensorflow.keras import layers
import pathlib
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import shutil


class TrainModel:

    def __init__(self):
        file_dir = dirname(__file__)
        self.dataPath = join(file_dir, "../data/glyphdataset/Dataset")
        self.manual = join(self.dataPath, "Manual/Preprocessed")
        self.automated = join(self.dataPath, "Automated/Preprocessed")
        self.convertedDataPath = join(file_dir, "../data/converteddataset")
        self.image_paths = []
        self.labels = []

        # Set dataset variables
        self.batch_size = 32
        self.img_height = 75
        self.img_width = 50
        self.num_classes = 179

        # Run the training program
        #self.download_dataset()
        #self.convert_dataset()
        self.load_dataset()
        self.normalize_images()
        self.normalize_images()
        self.set_prefetching()
        self.define_model()
        self.compile_model()
        self.start_training()
        self.show_training_accuracy()
        self.save_model()

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
        """
        Reorganize the dataset so each glyph class is in its own folder
        """
        if not exists(self.convertedDataPath):
            os.mkdir(self.convertedDataPath)

        # First convert the manually annotated dataset
        alldirs = [f for f in listdir(self.manual) if isdir(join(self.manual, f))]

        # Iterate over subfolders containing glyphs associated with a Piankoff photograph
        for dirs in alldirs:
            dir = self.manual + "/" + dirs
            if os.path.isdir(dir) == True:
                allfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

                # Iterate over glyph files
                for glyph in allfiles:

                    # Determine Gardiner designation
                    underscore_removed = glyph.split("_")
                    gardiner = underscore_removed[1].split(".")
                    glyph_location = dir + "/" + glyph

                    # Check if subfolder for Gardiner designation exist, if not it is created
                    if not exists(self.convertedDataPath + "/" + gardiner[0]):
                        os.mkdir(self.convertedDataPath + "/" + gardiner[0])

                    # Copy the PNG with glyph over to 'converteddataset/'
                    shutil.move(glyph_location, self.convertedDataPath + "/" + gardiner[0])

    def load_dataset(self):
        """
        Load the dataset from 'converteddataset
        """
        data_dir = pathlib.Path(self.convertedDataPath)

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(self.img_height, self.img_width),
          batch_size=self.batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(self.img_height, self.img_width),
          batch_size=self.batch_size)

    def normalize_images(self):
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.image_batch, self.labels_batch = next(iter(normalized_ds))
        first_image = self.image_batch[0]
        # Notice the pixels values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))

    def set_prefetching(self):
        """
        Set buffered prefetching to prevent I/O from blocking
        """
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def define_model(self):
        """
        Define the model to be trained
        """
        self.model = tf.keras.Sequential([
          layers.experimental.preprocessing.Rescaling(1./255),
          layers.Conv2D(32, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(self.num_classes)
        ])

    def compile_model(self):
        """
        Set optimizer and loss function
        """
        self.model.compile(
          optimizer='adam',
          loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

    def start_training(self):
        """
        Start model training
        """
        self.history = self.model.fit(
          self.train_ds,
          validation_data=self.val_ds,
          epochs=10
        )

    def show_training_accuracy(self):
        """
        Summarize history for accuracy and loss
        """
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def save_model(self):
        """
        Save the Keras model in SavedModel format
        """

        self.model.save("hieroglyph_model")
        img = self.image_batch[0]

        # print(img)
        #
        # img = (np.expand_dims(img,0))
        #
        # print(img.shape)
        #
        # predictions_single = self.model.predict(img)
        #
        # print(predictions_single)
        #
        # score = tf.nn.softmax(predictions_single[0])
        #
        # print(
        #     np.argmax(score), 100 * np.max(score)
        # )

if __name__ == "__main__":
    """
    Start the training application
    """
    app = TrainModel()
    app.download_dataset()
