import tensorflow as tf

model = tf.keras.models.load_model('hieroglyph_model')

# Check its architecture
# model.summary()

import numpy as np
from keras.preprocessing import image

img_height = 75
img_width = 50

# img = image.load_img('../data/glyphdataset/Dataset/Automated/Preprocessed/3/030026_S29.png', target_size = (img_height, img_width))
# img = image.load_img('../data/glyphdataset/Dataset/Automated/Preprocessed/3/030030_S29.png', target_size = (img_height, img_width))
img = image.load_img('../data/glyphdataset/Dataset/Automated/Preprocessed/3/030446_S29.png', target_size = (img_height, img_width))
# img = image.load_img('../data/glyphdataset/Dataset/Automated/Preprocessed/3/030024_E34.png', target_size = (img_height, img_width))
# img = image.load_img('../data/glyphdataset/Dataset/Automated/Preprocessed/3/030147_E34.png', target_size = (img_height, img_width))
# img = image.load_img('../data/glyphdataset/Dataset/Automated/Preprocessed/3/030443_E34.png', target_size = (img_height, img_width))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

gardiner = model.predict_classes(img)

print(gardiner)

# for i in range(len(img)):
# 	print("X=%s, Predicted=%s" % (img[i], gardiner[i]))

# Xnew = [[...], [...]]
ynew = model.predict(img)

# for i in range(len(img)):
# 	print("X=%s, Predicted=%s" % (img[i], ynew[i]))

classes = np.argmax(ynew, axis = 1)
print(classes)

# print(ynew)

