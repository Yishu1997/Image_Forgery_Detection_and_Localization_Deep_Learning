# Importing layers from Keras
 
from keras.layers import BatchNormalization 
from keras.layers import Activation
from keras.layers.convolutional import Conv2D

# Defining a convolution block with 2 convolution layers for UNet
 
def convolutional_2d_block(input_img, filters, kernel_size):
    
    x = Conv2D(filters = filters, kernel_initializer = 'he_normal',
               kernel_size = (kernel_size, kernel_size), padding = 'same')(input_img)
    
    # Convolutional layer followed by batch normalization and Relu layer
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    
    x = Conv2D(filters = filters, kernel_initializer = 'he_normal',
               kernel_size = (kernel_size, kernel_size), padding = 'same')(input_img)
    
    # Convolutional layer followed by batch normalization and Relu layer
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    
    return x
