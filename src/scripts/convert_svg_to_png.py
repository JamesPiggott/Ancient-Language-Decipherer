"""
A simple scripts that converts images in the HTML .svg to .png format.

The Hieroglyphics Initiative has 1072 glyphs in .svg format that they used to create their proper dataset by making people to trace over the image.

However, if they are in .png than perhaps they can be used as a baseline for a new dataset.

Requirement: pip install svglib

"""
import os

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


for file in os.listdir('../../examples/svg_hieroglyphs'):

    if '.svg' in file:
        filename = file
        filename = filename.split('.')[0]

        drawing = svg2rlg('../../examples/svg_hieroglyphs/' + file)

        if drawing is not None:
            renderPM.drawToFile(drawing, '../../examples/svg_hieroglyphs/' + filename + '.png', fmt='PNG')