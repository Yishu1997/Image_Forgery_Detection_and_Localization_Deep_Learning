# Importing libraries
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from ResNet50 import ResNet50 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

gpu_list = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_list[0], True)
print("Number of  GPUs: ", len(gpu_list))

# Model initialization
filters = np.array([64,64,256])
input_img, outputs =  ResNet50(filters = filters, classes = 2, input_shape = (256, 256, 3))
model = Model(inputs = [input_img], outputs = [outputs])

# Compiling the model
model.compile(optimizer = Adam(lr=0.002, epsilon=1e-08, decay=0.0), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Setting up checkpoint callback to save the model during training
checkpoint = ModelCheckpoint('C:/Keras_Tutorial/Casia/Entire_Casia_2/Saved_Models/ResNet-50/model_RNN_256by256input_BS32_Cr_LBP_Lr_point004_sparsecross_binarylabels_New_140_Epochs.model',
                                     verbose=2,
                                     monitor='val_loss',
		                             mode='min',
                                     save_weights_only=False,
                                     save_best_only=True)

# Train and Test data paths
train_path = 'C:/Keras_Tutorial/Casia/train_valid_combined/Cr_LBP'
test_path = 'C:/Keras_Tutorial/Casia/test/Cr_LBP'

# Batch size and target size
batch_size = 32
target_size = (256,256)

# Loading and augmenting data using ImageData Generator
trainloader = ImageDataGenerator(dtype = 'uint8', horizontal_flip=True,
    vertical_flip=True).flow_from_directory(directory = train_path, target_size = (256,256), classes = ['Fake', 'Real'], batch_size = 32, class_mode = 'binary')

# Augmentations are only applied to the train data

test_loader = ImageDataGenerator(dtype = 'uint8', ).flow_from_directory(directory = test_path, target_size = (256,256), classes = ['Fake', 'Real'], batch_size =32, class_mode = 'binary')
# Total number of epochs for training
epochs = 40

# Training the model
results =  model.fit_generator(trainloader, validation_data = test_loader, steps_per_epoch = trainloader.n//trainloader.batch_size, validation_steps = test_loader.n//test_loader.batch_size, epochs = epochs) #callbacks = [checkpoint]

# Accuracy and Loss graphs

figure, axes = plt.subplots(2, 1, figsize=(6, 6))
axes[0].plot(results.history['loss'], label="TrainLoss")
axes[0].plot(results.history['val_loss'], label="ValLoss")
axes[0].legend(loc='best', shadow=True)
axes[1].plot(results.history['acc'], label="TrainAcc")
axes[1].plot(results.history['val_acc'], label="ValAcc")
axes[1].legend(loc='best', shadow=True)
plt.show()
