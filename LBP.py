# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 07:48:26 2021

@author: yishu
"""

import skimage
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern
import cv2
import matplotlib.pyplot as plt
import os

image_path = 'C:/Keras_Tutorial/Casia/train/Cr/Real/Au_arc_30418.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype('float32')
orig_img =  plt.imread(image_path)

imgfilename = ''
outfilename = ''

# Input and output path
img_path = 'C:/Keras_Tutorial/Casia/Entire_Casia_2/Real/Cr'
out_path = 'C:/Keras_Tutorial/Casia/Entire_Casia_2/Real/Cr_LBP/Real/'
img = 0


# Loop to apply LBP to each image and save it in the respective directory
for dirname, _, filenames in os.walk(img_path):
    for filename in filenames:
        if filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            imgfilename = full_path
            filename_split = filename.split('.')
            #outfilename = os.path.join(out_path, filename_split[0] + '.mat')
            img = plt.imread(imgfilename)
            print(img.shape)
            LBP = local_binary_pattern(img, 8, 1)
            plt.imsave(fname = out_path + filename_split[0] + '.png', arr = LBP)
            #cv2.imwrite(out_path + filename_split[0] + '.png',Cr)

'''
#print(orig_img.shape)
#LBP = local_binary_pattern(orig_img, 8, 1)
#plt.imshow(LBP)
#LBP_DCT = cv2.dct(LBP)
#print(LBP_DCT)
#plt.imshow(LBP_DCT)
#print(plt_img.shape)
'''