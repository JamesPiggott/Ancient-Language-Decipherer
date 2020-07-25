from keras.applications.resnet50 import ResNet50

class TrainHieroglyphics:

    def __init__(self):
        print()

    def load_model(self):
        model = ResNet50()
        model.summary()