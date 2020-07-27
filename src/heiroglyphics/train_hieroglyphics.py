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

class TrainHieroglyphics:

    number_of_classes = None
    model = None
    image = None

    train_dir = ""
    validation_dir = ""

    def __init__(self):
        self.number_of_classes = 2
        self.train_dir = "../data/sample/train"
        self.validation_dir   = "../data/sample/validation"

    def run_all(self):
        self.load_model()
        self.add_layers_to_model()
        self.train_model()
        self.store_model()
        print("All done")

    def load_model(self):
        self.model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224,3))

        output = self.model.layers[-1].output
        output = Flatten()(output)

        self.model = Model(self.model.input, output)

        for layer in self.model.layers:
            layer.trainable = False
        
        self.model.summary()

    def add_layers_to_model(self):
        input_shape=(224, 224,3)
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

        train_datagen = ImageDataGenerator( rescale = 1.0/255. )
        test_datagen  = ImageDataGenerator( rescale = 1.0/255. )


        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            batch_size=int(32),
                                                            class_mode='binary',
                                                            target_size=(224, 224)) 

        validation_generator =  test_datagen.flow_from_directory(self.validation_dir,
                                                                batch_size=int(32),
                                                                class_mode  = 'binary',
                                                                target_size = (224, 224))



        history = self.model.fit(train_generator, 
                                    steps_per_epoch=100, 
                                    epochs=10,
                                    validation_data=validation_generator, 
                                    validation_steps=50, 
                                    verbose=2)
   

    def store_model(self):
        print("Store the model")
        self.model.save("test.h5")

    def perform_inference(self):
        features = self.model.predict(self.image)
        print("Features are: ")
        print(features.shape)