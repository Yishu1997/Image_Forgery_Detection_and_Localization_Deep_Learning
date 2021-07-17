# -*- coding: utf-8 -*-
"""
Created on Sat May  1 18:18:02 2021

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
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import losses
import albumentations as A

from clr_callback import CyclicLR

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_random_seed(1) 


original_height = 128
original_width = 128

transform = A.Compose([
    A.VerticalFlip(p=0.8),
    A.RandomRotate90(p=0.8),
    A.HorizontalFlip(p = 0.8),
    A.Transpose(p=0.8)
    ])

HFlip = A.HorizontalFlip(p=1)
VFlip = A.VerticalFlip(p=1)

# Set some parameters
im_width = 128
im_height = 128
border = 5
import cv2



# Training Data (Train + Valid)
train_ids = next(os.walk('C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP/Fake'))[2] # list of names all images in the given path
print("No. of Train images = ", len(train_ids))

X_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)

X_augmented = np.zeros((2 * len(train_ids), im_height, im_width, 1), dtype=np.float32)
y_augmented = np.zeros((2 * len(train_ids), im_height, im_width, 1), dtype=np.float32)

X_augmented_new = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
y_augmented_new = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)


# Load the images & masks into arrays
# tqdm is used to display the progress bar
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    # Load images
    #img = load_img("C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP/Fake/"+id_, grayscale=True)
    #x_img = img_to_array(img)
    
    x_img = cv2.imread('C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP/Fake/'+id_, cv2.IMREAD_GRAYSCALE)
    #print(x_img.shape)
    #x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    
    x_img = np.expand_dims(cv2.resize(x_img, (128, 128), interpolation = cv2.INTER_AREA), axis=2)
    # Load masks
    #mask = img_to_array(load_img("C:/Keras_Tutorial/Casia/CASIA_2_Groundtruth/"+id_.split('.')[0] + "_gt.png", grayscale=True))
    #mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
    # Save images
    #mask = np.expand_dims(cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1], axis=2)

    mask = cv2.imread("C:/Keras_Tutorial/Casia/CASIA_2_Groundtruth/"+id_.split('.')[0] + "_gt.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128), interpolation = cv2.INTER_AREA)
    mask = np.expand_dims(cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1], axis=2)
    X_train[n] = x_img/255.0
    y_train[n] = mask/255.0
    
    transformed = transform(image = X_train[n], mask = y_train[n])
    X_augmented_new[n] = transformed['image']
    y_augmented_new[n] = transformed['mask']
    

X_train = np.append(X_train, X_augmented_new, 0)
y_train = np.append(y_train, y_augmented_new, 0)

print('X_train Size: ', X_train.shape)
print('y_train Size: ', y_train.shape)

# Split train and valid
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Test Data
test_ids = next(os.walk('C:/Keras_Tutorial/Casia/test/Cr_LBP/Fake'))[2] # list of names all images in the given path
print("No. Validation/Test of images = ", len(test_ids))

X_valid = np.zeros((len(test_ids), im_height, im_width, 1), dtype=np.float32)
y_valid = np.zeros((len(test_ids), im_height, im_width, 1), dtype=np.float32)

# Load the images & masks into arrays
# tqdm is used to display the progress bar
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    # Load images
    #img = load_img("C:/Keras_Tutorial/Casia/test/Cr_LBP/Fake/"+id_, grayscale=True)
    #x_img = img_to_array(img)
    #print(x_img.shape)
    #x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    # Load masks
    #mask = img_to_array(load_img("C:/Keras_Tutorial/Casia/CASIA_2_Groundtruth/"+id_.split('.')[0] + "_gt.png", grayscale=True))
    #mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
    #mask = np.expand_dims(cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1], axis=2)
    
    x_img = cv2.imread('C:/Keras_Tutorial/Casia/test/Cr_LBP/Fake/'+id_, cv2.IMREAD_GRAYSCALE)    
    x_img = np.expand_dims(cv2.resize(x_img, (128, 128), interpolation = cv2.INTER_AREA), axis=2)

    mask = cv2.imread("C:/Keras_Tutorial/Casia/CASIA_2_Groundtruth/"+id_.split('.')[0] + "_gt.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128), interpolation = cv2.INTER_AREA)
    mask = np.expand_dims(cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1], axis=2)
    # Save images
    X_valid[n] = x_img/255.0
    y_valid[n] = mask/255.0

print('X_valid Size: ', X_valid.shape)
print('y_valid Size: ', y_valid.shape)


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

#input_layer = Input((im_height, im_width, 1))
#output_layer = build_model(input_layer, 32)

#model = Model(inputs=[input_layer], outputs=[output_layer])

model.compile(optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=[f1_MCC, matthews_correlation, IoU_try_mcc, "accuracy"]) # Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

checkpoint = ModelCheckpoint('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/CLR/Models/Journal_ANNA_New_Metrics_Drpot_False_Expansion_Unet_32Filters_Train+Valid_BS16_BCE_ADAM_ImgDataGenrtr_CLR_blr_1e-5_ml6_0005_ss_6_init_lr_1e-5_25Epochs.model',
		                             monitor='val_accuracy',
		                             verbose=2,
		                             save_best_only=True,
		                             mode='max',
		                             save_weights_only=False)     

# Image Data Generator
train_datagen = ImageDataGenerator()
#train_datagen.fit(X_train)
train_iterator = train_datagen.flow(X_train, y_train, batch_size= 16)

test_datagen = ImageDataGenerator()
test_iterator = test_datagen.flow(X_valid, y_valid, batch_size = 16)

#lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5 ,patience=7, min_lr = 0.00001, verbose = 1)

CLR = CyclicLR(
    mode='triangular',
    base_lr= 0.00001,
    max_lr=0.0007,
    step_size = 6* (X_train.shape[0] // 16) #STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE)
    )

epochs = 40

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#model.summary()

results = model.fit_generator(train_iterator, 
                              validation_data = test_iterator,
                              steps_per_epoch = train_iterator.n//train_iterator.batch_size,
                              validation_steps = test_iterator.n//test_iterator.batch_size,
                              epochs = epochs,
                              callbacks = [checkpoint, CLR])



# training loss and f1 plot
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, results.history["loss"], label="train_loss")
plt.plot(N, results.history["val_loss"], label="val_loss")
plt.plot(N, results.history["f1"], label="train_f1")
plt.plot(N, results.history["val_f1"], label="val_f1")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss/DSC")
plt.legend(loc="upper right")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/CLR/Plots/TP')


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
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/CLR/Plots/FP')

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
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/CLR/Plots/LP')


# learning rate history plot
N = np.arange(0, len(CLR.history["lr"]))
plt.figure()
plt.plot(N, CLR.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/CLR/Plots/CLRP')

