# PythonAncientLanguages
This Open Source project hopes to create a viable method to detect, recognize and interpret ancient written languages such as Egyptian Hieroglyphics, Sumerian Cuneiform or Mayan Glyphs. The method intended is Deep Learning with Convolutional layers using Tensorflow 2.x with its Keras API.

## Overview

Just like many other projects this is not without precedent. I first became aware of using Machine Learning to translate Hieroglyphs by reading the Master Thesis of Morris Franken (University of Amsterdam, 2013). Since that time the author has expanded on his work and created a Keras based version. The code base [glyphreader](https://github.com/morrisfranken/glyphreader) and his master thesis are in my opinion the best way to get started in this field. However, his work was hampered by the severely limited dataset which make training neural network models difficult. That is not to say it has not been attempted. F. Gimbert has attempted to use techniques from the field of face recognition to create a Deep Learning model to recognize glyphs. However, his project called [Hieroglyphs](https://github.com/fgimbert/Hieroglyphs) has been dormant since early 2019. Another attempt to classify singular hieroglyphic images was made by Barucci et al. (2021). Using the dataset of Morris Franken combined with additions of their own a high accuracy of 96 % was reached, but this is far away from translating images containing hieroglyph into modern text.

Since late 2017 there is also the Hieroglyphics Initiative from game publisher Ubisoft. This coincided with the release of their game Assassin's Creed Origins. Only recently did this initiative release a browser app (July 2020) called [Fabricius](https://artsexperiments.withgoogle.com/fabricius/en), in cooperation with Google. However, little of the project has been made public and neither is the dataset available. It is nonetheless useful for inspiration.

## Quickstart
Two scripts are currently 'mostly' done. The first allows manual feature extraction of images containing Hieroglyphs. The second trains a simple CNN-based model to recognize glyphs from the dataset collected by Morris Franken. Python 3 is required as are the packages in requirements.txt
```
pip install -r requirements.txt

python image_processing.py      # Feature extraction

python train_model.py           # Model training
```

## Project Goals
- Create a Deep Learning model for the detection and recognition of Egyptian Hieroglyphs. 
- Use a publicly available corpus for interpretation of such texts.
- Create an application that permits users to input a picture containing Ancient Egyptian texts and translate them, on the edge with no cloud computing. This application should have both an automated and manual process. The latter will make it possible to annotate datasets.

These are lofty goals with the focus being on Ancient Egyptian Hieroglyphs, for now. The planning is to proceed from image processing to glyph detection and then to recognition and finally interpretation and translation. The latter requires a corpus which do exist and is mainly an NLP problem. Where needed standard computer vision algorithms will be used. Even in 2022 the scientific literature emphasize their use for Hieroglyph extraction.

## Dataset
Finding annotated datasets on Ancient Egyptian Hieroglyphs is difficult. Besides the dataset assembled by Morris Franken there are no significant publicly available pools with which to train models. Nonetheless, belows is a list of small datasets. It remains to be seen if assembling them all together would improve training. Check the file public_datasets.md for details on these sources.
 - [Fayrose middle egyptian dataset](https://github.com/fayrose/MiddleEgyptianDictionaryWebsite)
 - [F Gimbert Hieroglyphs dataset](https://github.com/fgimbert/Hieroglyphs/tree/master/hieroglyphs)
 - [Google Arts Fabricius workbench dataset](https://github.com/googleartsculture/workbench/tree/main/src/assets/images)
 - [Jsesh dataset](https://github.com/rosmord/jsesh)

Note that the dataset from Google Arts is just one sample of each Hieroglyph. The dataset they collated for training, by asking people to trace over Hieroglyphs thus creating a set of some 50.000 in size, is still unavailable.

## References
- Franken, Morris & Gemert, Jan. (2013). Automatic Egyptian hieroglyph recognition by retrieving images as texts. MM 2013 - Proceedings of the 2013 ACM Multimedia Conference. 10.1145/2502081.2502199. 
- Domingo, Jaime & Herrera, Pedro & Valero, Enrique & Cerrada, Carlos. (2017). Deciphering Egyptian Hieroglyphs: Towards a New Strategy for Navigation in Museums. Sensors. 17. 589. 10.3390/s17030589. 
- Philipp Wiesenbach, & Stefan Riezler. (2019). Multi-Task Modeling of Phonographic Languages: Translating Middle Egyptian Hieroglyphs. Zenodo. http://doi.org/10.5281/zenodo.3524924
- Akshit Talwar. (2017). Script Identification with Ancient Egyptian Hieroglyphs. Dissertation from University of St Andrews. http://akshittalwar.com/files/dissertation.pdf
- Elnabawy, R., Elias, R., Salem, M.AM. et al. Extending Gardiner’s code for Hieroglyphic recognition and English mapping. Multimed Tools Appl 80, 3391–3408 (2021). https://doi.org/10.1007/s11042-020-09825-2
- A. Barucci, C. Cucci, M. Franci, M. Loschiavo and F. Argenti, A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification, in IEEE Access, vol. 9, pp. 123438-123447, 2021, doi: 10.1109/ACCESS.2021.3110082.

## Resources
- https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/