
from os.path import isdir, isfile, dirname

from os import listdir
from os.path import join


class StartApplication:

    def __init__(self):
        file_dir = dirname(__file__)
        self.dataPath = join(file_dir, "../data/glyphdataset/Dataset")
        # self.download_dataset()
        self.stelePath = join(self.dataPath, "Manual/Preprocessed")
        self.intermediatePath = join(file_dir, "../intermediates")
        self.featurePath = join(self.intermediatePath, "features.npy")
        self.labelsPath = join(self.intermediatePath, "labels.npy")
        self.svmPath = join(self.intermediatePath, "svm.pkl")
        self.image_paths = []
        self.labels = []
        self.batch_size = 200
        # self.model_training()


        # # Path to the data directory
        # data_dir = Path("../data/glyphdataset/Dataset/Manual/Preprocessed")
        #
        # # Get list of all the images
        # images = sorted(list(map(str, list(data_dir.glob("*.png")))))
        # labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
        # characters = set(char for label in labels for char in label)
        #
        # print("Number of images found: ", len(images))
        # print("Number of labels found: ", len(labels))
        # print("Number of unique characters: ", len(characters))
        # print("Characters present: ", characters)



        print("indexing images...")
        Steles = [join(self.stelePath, f) for f in listdir(self.stelePath) if isdir(join(self.stelePath, f))]
        for stele in Steles:
            count = 0
            imagePaths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]
            for path in imagePaths:
                print(path)
                count += 1
            print(count)
    #         self.image_paths.append(path)
    #         self.labels.append(path[(path.rfind("_") + 1): path.rfind(".")])
    #
    # print("computing features...")
    # for idx, (batch_images, _) in enumerate(batchGenerator(self.image_paths, self.labels, self.batch_size)):
    #     print("{}/{}".format((idx + 1) * self.batch_size, len(self.labels)))

if __name__ == "__main__":
    """
    Start the application
    """
    app = StartApplication();