"""
This little script creates a new subdirectory for each unique Hieroglyph.

This ensures conformity with the TensorFlow 2.0 API.

"""

import os

from shutil import copyfile


for file in os.listdir('../../examples/svg_hieroglyphs'):

    if '.png' in file:
        filename = file.split('.')[0]

        if not os.path.exists('../../examples/svg_hieroglyphs/' + filename):
            os.mkdir('../../examples/svg_hieroglyphs/' + filename)

        copyfile('../../examples/svg_hieroglyphs/' + file, '../../examples/svg_hieroglyphs/' + filename + "/" + file)