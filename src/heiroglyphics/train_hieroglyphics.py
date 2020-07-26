from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

class TrainHieroglyphics:

    number_of_classes = 171
    model = None
    image = None

    def __init__(self):
        self.load_model()
        self.load_image()
        self.perform_inference()
        print("All done")

    def load_model(self):
        model = ResNet50(include_top=False, input_shape=(300, 300, 3))

        flat1 = Flatten()(model.outputs)
        class1 = Dense(1024, activation='relu')(flat1)
        output = Dense(self.number_of_classes, activation='softmax')(class1)

        self.model = Model(inputs=model.inputs, outputs=output)

        model.summary()


    def load_image(self):
        image = load_img('dog.jpg', target_size=(224, 224))

        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        self.image = preprocess_input(image)

    def perform_inference(self):
        features = self.model.predict(self.image)
        print(features.shape)