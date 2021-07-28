# Importing Libraries

import os
import cv2

# Input and Output file names
imgfilename = ''
outfilename = ''

# Input and Output paths
img_path = 'C:/Keras_Tutorial/CASIA Full/CASIA2.0_revised/Au'
out_path = 'C:/Keras_Tutorial/CASIA Full/CASIA2.0_revised/Cr/Au/'

# Loop to convert to YCbCr color space and extract Cr channel
for dirname, _, filenames in os.walk(img_path):
    for filename in filenames:
        if filename.endswith('tif') or filename.endswith('jpg') or filename.endswith('bmp') or filename.endswith('JPG'):
            
            full_path = os.path.join(dirname, filename)
            imgfilename = full_path
            filename_split = filename.split('.')
            
            # Reading image from directory
            img = cv2.imread(imgfilename)
            b,g,r = cv2.split(img)
            img_rgb = cv2.merge((r,g,b))
            
            #Coverting the RGB image to YCrCb color space and extracting Cr channel
            img_YCrCb =cv2.cvtColor(img.copy(), cv2.COLOR_RGB2YCrCb)
            Y,Cr,Cb = cv2.split(img_YCrCb)
            
            # Saving Cr image to directory
            cv2.imwrite(out_path + filename_split[0] + '.png',Cr)
            