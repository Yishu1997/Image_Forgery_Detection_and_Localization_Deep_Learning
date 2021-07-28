# Importing Libraries

from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, Dense, Flatten, MaxPooling2D, AveragePooling2D, Input
from tensorflow.keras.initializers import glorot_uniform

from Identity_Block_ResNet50 import identity_2d_block
from Convolutional_Block_ResNet50 import convolutional_2d_block

# Defining a function for the ResNet-50 model architecture

def ResNet50(filters, classes = 2, input_shape = (256, 256, 3)):
    """
    
    Parameters
    ----------
    filters : numpy array of number of filters for identity as well as convolutional blocks
    classes : Totsl number of classes (labels)
    input_shape : shape of the input to the ResNet-50 model (channels last)
    
    Returns
    -------
    input_img : input to the model
    outputs: The output of the model
    
    """
    
    # Input tensor
    input_img = Input(input_shape)
    
    # (3,3)  Zero Padding the input
    x = ZeroPadding2D(padding = 3)(input_img)
    
    # First Convolutional layers with (7,7) kernel
    x = Conv2D(filters = 64, kernel_size = 7, strides = 2, padding = "valid", kernel_initializer = glorot_uniform(0))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)
    
    # 1 convolutional block followed by 2 identity blocks
    x = convolutional_2d_block(x, filters = filters, kernel_size = 3, strides = 1)
    x = identity_2d_block(x, filters = filters, kernel_size = 3)
    x = identity_2d_block(x, filters = filters, kernel_size = 3)
    
    # 1 convolutional block followed by 3 identity blocks
    x = convolutional_2d_block(x, filters = filters * 2, kernel_size = 3, strides = 2)
    x = identity_2d_block(x, filters = filters * 2, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 2, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 2, kernel_size = 3)
    
    # 1 convolutional block followed by 5 identity blocks
    x = convolutional_2d_block(x, filters = filters * 4, kernel_size = 3, strides = 2)
    x = identity_2d_block(x, filters = filters * 4, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 4, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 4, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 4, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 4, kernel_size = 3)
    
    # 1 convolutional blocks followed by 2 identity blocks
    x = convolutional_2d_block(x, filters = filters * 8, kernel_size = 3, strides = 2)
    x = identity_2d_block(x, filters = filters * 8, kernel_size = 3)
    x = identity_2d_block(x, filters = filters * 8, kernel_size = 3)
    
    # 2D Average Pooling Layer before flattening
    x = AveragePooling2D(pool_size = 2)(x)
    
    # Falttening and output layer with softmax
    x = Flatten()(x)
    outputs = Dense(activation = "softmax", units = classes, kernel_initializer = glorot_uniform(0))(x)
    
    return input_img, outputs