# PythonAncientLanguages

This Open Source project hopes to create a viable method to detect, recognize and interpret ancient written languages such as Egyptian Hieroglyphics, Sumerian Cuneiform or Mayan Glyphs. The method intended is Deep Learning with Convolutional layers using Tensorflow 2.x with the Keras API.

Just like many other projects this is not without precedent. I first became aware of using Machine Learning to translate Hieroglyphs by reading the Master Thesis of Morris Franken (University of Amsterdam). Since that time the author has expanded on his work and created a Keras based version. The code base [GitHub](https://github.com/morrisfranken/glyphreader) and his master thesis are in my opinion the best way to get started in this field. However, his work was hampered by the severely limited dataset available.

Since late 2017 there is also the Hieroglyphics Initiative from game publisher Ubisoft. This coincided with the release of their game Assassin's Creed Origins. Only recently did this initiative release a Desktop app (July 2020), in cooperation with Google. However, little of the project has been made public and neither is the dataset available.

## Project Goals
- Create Deep Learning models for the detection and recognition of Egyptian Hieroglyphs and interpretation of texts containing them.
- Create an application that permits users to input a picture containing Ancient Egyptian texts and translate them.

These are lofty goals with the focus being on Ancient Hieroglyphs. The planning is to proceed from glyph detection to recognition and finally interpretation and translation. The latter requires corpus which do exist and is mainly an NLP problem

## Short-term goals
- Reproduce the work of Morris Franken & Jan Gemert.
   - The purpose of this to get a proper understanding how Hieroglyph detection would work and to make use of their tooling
   - Attempt to update their work to 2021 standards
   - Create a download location for their data (as a backup).
- Reproduce the work of Domingo, Herrera, Valero and Cerrada.

## Necessities

After performing a literature study (see list of references) as well as studying other code projects I have identified the following necessities for continuing this project. Feel free to make suggestions as to how they can be attained.

 - Obtain a copy of "The Pyramid of Unas" by Alexandre Piankoff, published 1955 or 1969. This book I could use to complete the dataset collated by Morris Franken.
 - Obtain a copy of the Abydos King's list dataset.

## References
- Franken, Morris & Gemert, Jan. (2013). Automatic Egyptian hieroglyph recognition by retrieving images as texts. MM 2013 - Proceedings of the 2013 ACM Multimedia Conference. 10.1145/2502081.2502199. 
- Domingo, Jaime & Herrera, Pedro & Valero, Enrique & Cerrada, Carlos. (2017). Deciphering Egyptian Hieroglyphs: Towards a New Strategy for Navigation in Museums. Sensors. 17. 589. 10.3390/s17030589. 
- Philipp Wiesenbach, & Stefan Riezler. (2019). Multi-Task Modeling of Phonographic Languages: Translating Middle Egyptian Hieroglyphs. Zenodo. http://doi.org/10.5281/zenodo.3524924
- Akshit Talwar. (2017). Script Identification with Ancient Egyptian Hieroglyphs. Dissertation from University of St Andrews. http://akshittalwar.com/files/dissertation.pdf

## Resources
- https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/