import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import tensorflow as tf
from tensorflow.keras import layers
import pathlib

# Location of the hieroglyph dataset, each class divided into subfolders
data_dir = "../train_data"
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 75
img_width = 50

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Normalize the images
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Set buffered prefetching to prevent I/O from blocking
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 179


img = image_batch[0]

print(img)

img = (np.expand_dims(img,0))

print(img.shape)
