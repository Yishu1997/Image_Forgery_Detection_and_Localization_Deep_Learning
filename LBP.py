# Importing Libraries 

from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os

# Input and Output file names
imgfilename = ''
outfilename = ''

# Input and output path
img_path = 'C:/Keras_Tutorial/CASIA Full/CASIA2.0_revised/Cr/Au'
out_path = 'C:/Keras_Tutorial/CASIA Full/CASIA2.0_revised/Cr+LBP/Train/Au/'
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
            #print(img.shape)
            LBP = local_binary_pattern(img, 8, 1)
            plt.imsave(fname = out_path + filename_split[0] + '.png', arr = LBP)
