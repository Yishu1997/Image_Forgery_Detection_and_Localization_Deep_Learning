# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:17:53 2021

@author: yishu
"""

import os
import cv2
import numpy as np
import pandas as pd 

list_of_image_files = []
list_of_dirs_img =  os.listdir('C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP/Fake')
list_of_dirs_masks = os.listdir('C:/Keras_Tutorial/Casia/train_valid_groundtruth/Groundtruth')
print(len(list_of_dirs_masks))
act_images = {}
binary_mask = {}

img_mask_dim = (128, 128)
list_ids = []
for i in range(len(list_of_dirs_img)):
    key = list_of_dirs_img[i].split('_')[7].split('.')[0]
    list_ids.append(key)
    rgb_img_data = cv2.imread('C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP/Fake/'+ list_of_dirs_img[i], cv2.IMREAD_GRAYSCALE)
    rgb_img_data = cv2.resize(rgb_img_data, (128, 128), interpolation = cv2.INTER_AREA)
    mask = cv2.imread('C:/Keras_Tutorial/Casia/train_valid_groundtruth/Groundtruth/'+ list_of_dirs_img[i].split('.')[0] + '_gt.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128), interpolation = cv2.INTER_AREA)
    #print("Image shape: ", rgb_img_data.shape)
    #print('Mask shape: ', mask.shape)
    act_images[key] = rgb_img_data
    binary_mask[key] = mask
    


np.save("C:/Keras_Tutorial/Casia/train_valid_combined/npy_files_with small_forgery_images_removed/train+valid_small_forgery_removed_images.npy", act_images)
np.save("C:/Keras_Tutorial/Casia/train_valid_combined/npy_files_with small_forgery_images_removed/train+valid_small_forgery_removed_binary_mask.npy", binary_mask)

import csv

df = pd.DataFrame(dict(ids=list_ids))
df.ids = df.ids.apply('="{}"'.format)
df.to_csv('train+valid_small_forgery_removed_ids.csv')
'''with open('IDS.csv', 'w', newline ='') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
    for i in range(len(list_ids)):
        write.writerow([list_ids[i]])'''

