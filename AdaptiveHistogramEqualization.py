# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:28:04 2021

@author: yishu
"""

import cv2
import matplotlib.pyplot as plt
import os

#image_path = 'C:/Keras_Tutorial/Casia/train/Noiseprint_train/Fake/Tp_D_NRN_S_B_art00087_art00087_01010.png'

image_path = 'C:/Keras_Tutorial/Casia/test/Noiseprint_Fake/Noiseprint_Output/'
output_path = 'C:/Keras_Tutorial/Casia/test/Noiseprint_AHE/Fake/'

# Loop for applying AHE to Noiseprint images
for dirname, _, filenames in os.walk(image_path):
    #print("DIR: ",dirname)
    #print("_____", _)
    #print("Filenames", filenames)
    for filename in filenames:
        if filename.endswith('tif') or filename.endswith('jpg') or filename.endswith('bmp') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            imgfilename = full_path
            filename_split = filename.split('.')

            image = cv2.imread(imgfilename)
            #b,g,r = cv2.split(img)
            #img_rgb = cv2.merge((r,g,b))
            # Converting BGR to RGB Image
            image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            # Converting RGB to Grayscale Image
            image_grayscale = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2GRAY)
            
            # Adaptive Histogram Equalization            
            clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (2,2))
            image_AHE = clahe.apply(image_grayscale.copy())
            #plt.imshow(image_ADE)
            
            # Converting equalized image to RGB
            image_AHE_rgb = cv2.cvtColor(image_AHE.copy(), cv2.COLOR_GRAY2RGB)
            cv2.imwrite(output_path + filename_split[0] + '.png', image_AHE_rgb)
            #print('Working')

'''
image  = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB) 
#plt.imshow(image)
#plt.imshow(image_rgb)

image_grayscale = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2GRAY)
print(image_grayscale.shape)
print(image_rgb.shape)
#plt.imshow(image_grayscale)

clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
image_ADE = clahe.apply(image_grayscale.copy())

#plt.imshow(image_ADE)

image_ADE_rgb = cv2.cvtColor(image_ADE.copy(), cv2.COLOR_GRAY2RGB)
plt.imshow(image_ADE_rgb)
'''
