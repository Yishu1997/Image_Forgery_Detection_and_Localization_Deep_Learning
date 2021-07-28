# Importing Keras layers

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.initializers import glorot_uniform

# Function to define the Identity block with 3 convolutional layers

def convolutional_2d_block(input_img, filters, kernel_size = 3, strides = 2):
    """
    
    Parameters
    ----------
    input_img : input tensor (channels last)
    filters : list of filters for the 3 convolutional layers
    kernel_size : filter size of the 2nd convolutional layer
    strides: value of the stride to be used
    """
    
    # Number of filters for each convolutional layers
    filter1, filter2, filter3 = filters
    
    # Input to the shortcut connection
    x_shortcut = input_img
    
    # Convolutional Layer 1
    x = Conv2D(kernel_size = 1, filters = filter1, strides = strides, kernel_initializer = glorot_uniform(seed = 0), padding = "valid")(input_img) # seed of glorot uniform initializer is set to 0 for random initialization
    x = BatchNormalization(axis = 3)(x)  # axis is set to 3 as we use the channel last scheme
    x = Activation('relu')(x)
    
    # Convolutional Layer 2
    x = Conv2D(kernel_size = kernel_size, filters = filter2, strides = 1, kernel_initializer = glorot_uniform(seed = 0), padding = "same")(x)
    x = BatchNormalization(axis = 3)(x)  
    x = Activation('relu')(x)
    
    # Covolutional layer 3
    x = Conv2D(kernel_size = 1, filters = filter3, strides = 1, kernel_initializer = glorot_uniform(seed = 0), padding = "valid")(x)
    x = BatchNormalization(axis = 3)(x)  
    
    # Shortcut connection for convolutional block
    x_shortcut = Conv2D(kernel_size = 1, filters = filter3, strides = strides, kernel_initializer = glorot_uniform(seed = 0), padding = "valid")(x_shortcut)
    x_shortcut = BatchNormalization(axis = 3)(x_shortcut)
    
    # Adding the shortcut connection and then applying the activation
    x = Add()([x_shortcut, x])
    x = Activation('relu')(x)
    
    return x