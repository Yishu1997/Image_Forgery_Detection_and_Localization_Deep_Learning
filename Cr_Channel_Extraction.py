import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

imgfilename = ''
outfilename = ''

img_path = 'C:/Keras_Tutorial/Casia/Entire_Casia_2/Real_Original'
out_path = 'C:/Keras_Tutorial/Casia/Entire_Casia_2/Real/Cr/'

# Loop to convert to YCbCr color space and extract Cr channel
for dirname, _, filenames in os.walk(img_path):
    #print("DIR: ",dirname)
    #print("_____", _)
    #print("Filenames", filenames)
    for filename in filenames:
        if filename.endswith('tif') or filename.endswith('jpg') or filename.endswith('bmp') or filename.endswith('JPG'):
            full_path = os.path.join(dirname, filename)
            imgfilename = full_path
            filename_split = filename.split('.')
            #outfilename = os.path.join(out_path, filename_split[0] + '.mat')
            img = cv2.imread(imgfilename)
            b,g,r = cv2.split(img)
            img_rgb = cv2.merge((r,g,b))
            img_YCrCb =cv2.cvtColor(img.copy(), cv2.COLOR_RGB2YCrCb)
            Y,Cr,Cb = cv2.split(img_YCrCb)
            cv2.imwrite(out_path + filename_split[0] + '.png',Cr)
            #noiseprint_filename ='C:/Keras_Tutorial/Casia/train/Noiseprint_Fake/TestNew/' + filename_split[0] + '.png'
            #print("ERROR 3")
            #plt.imsave(fname = noiseprint_filename, arr = res.clip(vmin,vmax),vmin = vmin, vmax = vmax)
