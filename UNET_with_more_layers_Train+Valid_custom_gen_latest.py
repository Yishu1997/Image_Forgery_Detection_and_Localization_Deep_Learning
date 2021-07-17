# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:51:16 2021

@author: yishu
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

from tqdm import tqdm, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import losses
import albumentations as A

import keras.backend as K

from custom_gen_latest import CustomDataGen

from clr_callback import CyclicLR

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Set some parameters
im_width = 128
im_height = 128


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # Convolution Block: 2 convolutional layers with the parameters passed to it
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # Defining my UNET Model
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    #p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    #p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    #p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    #p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    #p5 = Dropout(dropout)(p5)
    
    c6 = conv2d_block(p5, n_filters = n_filters * 32, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u7 = Conv2DTranspose(n_filters * 16, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c5])
    #u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c4])
    #u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c3])
    #u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u10 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u10 = concatenate([u10, c2])
    #u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u11 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c10)
    u11 = concatenate([u11, c1])
    #u11 = Dropout(dropout)(u11)
    c11 = conv2d_block(u11, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def IoU_try_mcc(y_true, y_pred):
    
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    return K.mean(tp / (tp + fp + fn))

def f1_MCC(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True)
model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy',f1_MCC, matthews_correlation])

checkpoint = ModelCheckpoint('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Models/CLR_Models/Anna_rmovd_smallforged_Unet_32filters_Drput_False_BS32_BCE_Adam_Lr_1e-5_CLR_triangular_blr_1e-5_mlr_0007_ss_6_custom_gen_latest_40Epochs.model',
		                             monitor='val_loss',
		                             verbose=2,
		                             save_best_only=True,
		                             mode='min',
		                             save_weights_only=False)

#lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5 ,patience=10, min_lr = 0.00001, verbose = 1)

batch_size = 32

train_df = pd.read_csv("C:\\Keras_Tutorial\\train+valid_small_forgery_removed_ids.csv")
train_df = train_df.sample(frac=1).reset_index(drop=True)

act_images = np.load("C:/Keras_Tutorial/Casia/train_valid_combined/npy_files_with small_forgery_images_removed/train+valid_small_forgery_removed_images.npy", allow_pickle=True).item()
binary_mask = np.load("C:/Keras_Tutorial/Casia/train_valid_combined/npy_files_with small_forgery_images_removed/train+valid_small_forgery_removed_binary_mask.npy", allow_pickle=True).item()

valid_df = pd.read_csv("C:\\Keras_Tutorial\\test_ids.csv")
valid_df = valid_df.sample(frac=1).reset_index(drop=True)

valid_act_images = np.load("C:/Keras_Tutorial/Casia/test/test_act_images.npy", allow_pickle=True).item()
valid_binary_mask = np.load("C:/Keras_Tutorial/Casia/test/test_binary_mask.npy", allow_pickle=True).item()

traingen = CustomDataGen(train_df, act_images, binary_mask, X_col={'ids':'ids'},
                         batch_size=batch_size, ftrain_data=True)

validgen = CustomDataGen(valid_df, valid_act_images, valid_binary_mask, X_col={'ids':'ids'},
                         batch_size=batch_size, ftrain_data=False)

CLR = CyclicLR(
    mode='triangular',
    base_lr= 0.00001,
    max_lr=0.0007,
    step_size = 6 * (traingen.n // traingen.batch_size) #STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE)
    )

epochs = 40

results = model.fit_generator(traingen, 
                              validation_data = validgen,
                              steps_per_epoch = traingen.n//traingen.batch_size,
                              validation_steps = validgen.n//validgen.batch_size,
                              epochs = epochs,
                              callbacks = [checkpoint, CLR])


# training loss and F1 plot
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, results.history["loss"], label="train_loss")
plt.plot(N, results.history["val_loss"], label="val_loss")
plt.plot(N, results.history["f1"], label="train_f1")
plt.plot(N, results.history["val_f1"], label="val_f1")
plt.title("Training Loss and F1")
plt.xlabel("Epoch #")
plt.ylabel("Loss/F1")
plt.legend(loc="upper right")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Plots/CLR_Plots/TP')

# learning rate history plot
N = np.arange(0, len(CLR.history["lr"]))
plt.figure()
plt.plot(N, CLR.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Plots/CLR_Plots/CLRP')

# F1 plot
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, results.history["f1"], label="train_f1")
plt.plot(N, results.history["val_f1"], label="val_f1")
plt.title("Training F1")
plt.xlabel("Epoch #")
plt.ylabel("F1")
plt.legend(loc="upper right")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Plots/FP')

# Loss Plot
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, results.history["loss"], label="train_loss")
plt.plot(N, results.history["val_loss"], label="val_loss")
plt.title("Training Loss and DSC")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Plots/LP')