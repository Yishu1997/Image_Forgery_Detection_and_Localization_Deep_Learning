# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:04:30 2021

@author: yishu
"""
import numpy as np
import math 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Identity_block
def identity_block(X, f, filters, stage, block):
    """
    Implementing the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value (shortcut). Will need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding ='same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Adding shortcut value to the main path, and passing it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X

# Convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementing the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Saving the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c',kernel_initializer = glorot_uniform(seed=0))(X) 
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

# ResNet50

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FULLYCONNECTEDLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    
    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'a', s =2)
    X = identity_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'b')
    X = identity_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'c')
    X = identity_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'd')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s =2)
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'b')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'c')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'd')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'e')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', s =2)
    X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'b')
    X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'c')

    # AVGPOOL
    X = AveragePooling2D((2,2), name = 'avg_pool')(X)
    

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

# Making an instance of the ResNet50 model

model = ResNet50(input_shape = (256, 256, 3), classes = 2)
#model.summary()

# Compiling the model
model.compile(optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Importing Data using Image Data Generator

train_path = 'C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP'
valid_path = 'C:/Keras_Tutorial/Casia/test/Cr_LBP'
test_path = 'C:/Keras_Tutorial/Casia/test/Cr_LBP'
noiseprint_path = 'C:/Keras_Tutorial/Casia/train/Noiseprint_train'

 
trainloader = ImageDataGenerator(dtype = 'uint8', horizontal_flip=True,
    vertical_flip=True).flow_from_directory(directory = train_path, 
                                                         target_size = (256,256),
                                                         classes = ['Fake', 'Real'],
                                                         batch_size = 32,
                                                         class_mode = 'binary'
                                                         )

validloader= ImageDataGenerator(dtype = 'uint8', ).flow_from_directory(directory = valid_path, 
                                                         target_size = (256,256),
                                                         classes = ['Fake', 'Real'],
                                                         batch_size =32,
                                                         class_mode = 'binary'
                                                         )
test_batches = ImageDataGenerator(dtype = 'uint8').flow_from_directory(directory = test_path, 
                                                         target_size = (256,256),
                                                         classes = ['Fake', 'Real'],
                                                         batch_size = 32,
                                                         shuffle = False,
                                                         class_mode = 'binary'
                                                         )
'''noiseprint_batches = ImageDataGenerator(dtype = 'uint8', featurewise_std_normalization = True).flow_from_directory(directory = noiseprint_path, 
                                                         target_size = (224,224),
                                                         classes = ['Fake', 'Real'],
                                                         batch_size = 2,
                                                         class_mode = 'binary'
                                                         )'''

                                            
checkpoint = ModelCheckpoint('C:/Keras_Tutorial/Casia/Entire_Casia_2/Saved_Models/ResNet-50/model_RNN_256by256input_BS32_Cr_LBP_Lr_point004_sparsecross_binarylabels_New_140_Epochs.model',
		                             monitor='val_loss',
		                             verbose=2,
		                             save_best_only=True,
		                             mode='min',
		                             save_weights_only=False)
callbacks_list = [checkpoint]
                                            
x,y = next(trainloader)  
print(x)            
#model.summary()                             
results = model.fit_generator(trainloader,
                    validation_data = validloader,
                    steps_per_epoch = trainloader.n//trainloader.batch_size,
                    validation_steps = validloader.n//validloader.batch_size,
                    epochs=50)
"""
X_train, Y_train = next(train_batches)
X_valid, Y_valid = next(valid_batches)
print(Y_train)

model.fitg(X_train, Y_train, epochs = 20, batch_size = 10)

#model.save('C:/Keras_Tutorial/My_ResNet50')

pred = model.evaluate(X_valid, Y_valid)

print("Loss = ", pred[0])
print("Validation Accuracy", pred[1])
"""

fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(results.history['loss'], label="TrainLoss")
ax[0].plot(results.history['val_loss'], label="ValLoss")
ax[0].legend(loc='best', shadow=True)
ax[1].plot(results.history['acc'], label="TrainAcc")
ax[1].plot(results.history['val_acc'], label="ValAcc")
ax[1].legend(loc='best', shadow=True)
plt.show()
