# Importing libraries

from keras.layers.convolutional import Conv2DTranspose, Conv2D
from Convolutional_2D_Block import convolutional_2d_block
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

# Defining the modifeid UNet model architecture

def Modified_UNet(input_img, filters = 32, pool_size = (2,2)):
    """
    Parameters
    ----------
    input_img : input tensor
    
    filters : Initial number of filters.The default is 32.
    
    pool_size : The pool size for the pooling layers.The default is (2,2).

    Returns
    -------
    input_img : input tensor
    
    outputs: The segmentation output of the model

    """
    
    # Contracting path of the modified UNet
    
    conv1 = convolutional_2d_block(input_img, kernel_size = 3, filters = filters * 1)
    max_pool1 = MaxPooling2D((pool_size))(conv1)
    
    conv2 = convolutional_2d_block(max_pool1, kernel_size = 3, filters = filters * 2)
    max_pool2 = MaxPooling2D((pool_size))(conv2)
    
    conv3 = convolutional_2d_block(max_pool2, kernel_size = 3, filters = filters * 4)
    max_pool3 = MaxPooling2D((pool_size))(conv3)
    
    conv4 = convolutional_2d_block(max_pool3, kernel_size = 3, filters = filters * 8)
    max_pool4 = MaxPooling2D((pool_size))(conv4)
    
    conv5 = convolutional_2d_block(max_pool4, kernel_size = 3, filters = filters * 16)
    max_pool5 = MaxPooling2D((pool_size))(conv5)
    
    conv6 = convolutional_2d_block(max_pool5, kernel_size = 3, filters = filters * 32)
    
    # Expanding path of the modified UNet
    
    upconv7 = Conv2DTranspose(kernel_size = 3, filters = filters * 16, padding = 'same', strides = (2,2))(conv6)
    upconv7 = concatenate([upconv7, conv5])
    conv7 = convolutional_2d_block(upconv7, kernel_size = 3, filters = filters * 16)
    
    upconv8 = Conv2DTranspose(kernel_size = 3, filters = filters * 8, padding = 'same', strides = (2,2))(conv7)
    upconv8 = concatenate([upconv8, conv4])
    conv8 = convolutional_2d_block(upconv8, kernel_size = 3, filters = filters * 8)
    
    upconv9 = Conv2DTranspose(kernel_size = 3, filters = filters * 4, padding = 'same', strides = (2,2))(conv8)
    upconv9 = concatenate([upconv9, conv3])
    conv9 = convolutional_2d_block(upconv9, kernel_size = 3, filters = filters * 4)
    
    upconv10 = Conv2DTranspose(kernel_size = 3, filters = filters * 2, padding = 'same', strides = (2,2))(conv9)
    upconv10 = concatenate([upconv10, conv2])
    conv10 = convolutional_2d_block(upconv10, kernel_size = 3, filters = filters * 2)
    
    upconv11 = Conv2DTranspose(kernel_size = 3, filters = filters * 1, padding = 'same', strides = (2,2))(conv10)
    upconv11 = concatenate([upconv11, conv1])
    conv11 = convolutional_2d_block(upconv11, kernel_size = 3, filters = filters * 1)

    outputs = Conv2D(1, kernel_size = 1, activation = 'sigmoid')(conv11)

    return input_img, outputs