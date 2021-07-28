# Importing Libraries

import cv2
import os

# Input and Output Paths

image_path = 'C:/Keras_Tutorial/Casia/test/Noiseprint_Fake/Noiseprint_Output/'
output_path = 'C:/Keras_Tutorial/Casia/test/Noiseprint_AHE/Fake/'

# Loop for applying AHE to Noiseprint images

for dirname, _, filenames in os.walk(image_path):
    for filename in filenames:
        if filename.endswith('tif') or filename.endswith('jpg') or filename.endswith('bmp') or filename.endswith('png') or filename.endswith('JPG'):
            
            full_path = os.path.join(dirname, filename)
            imgfilename = full_path
            filename_split = filename.split('.')
            
            # Reading Image from Directory
            image = cv2.imread(imgfilename)
            image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image_grayscale = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2GRAY)
            
            # Adaptive Histogram Equalization            
            clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (2,2))
            image_AHE = clahe.apply(image_grayscale.copy())
            
            
            # Saving AHE image to directory
            image_AHE_rgb = cv2.cvtColor(image_AHE.copy(), cv2.COLOR_GRAY2RGB)
            cv2.imwrite(output_path + filename_split[0] + '.png', image_AHE_rgb)
