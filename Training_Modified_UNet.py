# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from Modified_UNet import Modified_UNet
from Segmentation_Metrics import f1_score, MCC
from custom_gen_latest import CustomDataGen
from clr_callback import CyclicLR

# Input Width and Height

im_width = 128
im_height = 128

# Batch Size
batch_size = 32

# Model Input
input_img = Input((im_height, im_width, 1))

# Model Initialization
input_img, outputs = Modified_UNet(input_img, filters = 32, pool_size = (2,2))
model = Model(inputs = [input_img], outputs = [outputs])

# Compiling the model
model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 
              loss='binary_crossentropy', metrics=['accuracy',f1_score, MCC])

# Setting up checkpoint callback to save the model during training
checkpoint = ModelCheckpoint('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Models/CLR_Models/Anna_rmovd_smallforged_Unet_32filters_Drput_False_BS32_BCE_Adam_Lr_1e-5_CLR_triangular_blr_1e-5_mlr_0007_ss_6_custom_gen_latest_40Epochs.model',
		                             monitor='val_loss',
		                             verbose=2,
		                             save_best_only=True,
		                             mode='min',
		                             save_weights_only=False)

# Loading data
train_df = pd.read_csv("C:\\Keras_Tutorial\\train_images_ids.csv")
train_df = train_df.sample(frac=1).reset_index(drop=True)

train_images = np.load("C:/Keras_Tutorial/Casia/train_valid_combined/npy_files/train_images.npy", allow_pickle=True).item()
train_binary_mask = np.load("C:/Keras_Tutorial/Casia/train_valid_combined/npy_files/binary_mask.npy", allow_pickle=True).item()

test_df = pd.read_csv("C:\\Keras_Tutorial\\test_ids.csv")
test_df = test_df.sample(frac=1).reset_index(drop=True)

test_images = np.load("C:/Keras_Tutorial/Casia/test/test_act_images.npy", allow_pickle=True).item()
test_binary_mask = np.load("C:/Keras_Tutorial/Casia/test/test_binary_mask.npy", allow_pickle=True).item()

# Instances of Custom Datat Generator
traingen = CustomDataGen(train_df, train_images, train_binary_mask, X_col={'ids':'ids'},
                         batch_size=batch_size, ftrain_data=True)

validgen = CustomDataGen(test_df, test_images, test_binary_mask, X_col={'ids':'ids'},
                         batch_size=batch_size, ftrain_data=False)

# Cyclic Learning Rate Callback 
CLR = CyclicLR(
    mode='triangular',
    base_lr= 0.00001,
    max_lr=0.0007,
    step_size = 6 * (traingen.n // traingen.batch_size) 
    )

# Total number of epochs for training 
epochs = 40

# Training the model
results = model.fit_generator(traingen, 
                              validation_data = validgen,
                              steps_per_epoch = traingen.n//traingen.batch_size,
                              validation_steps = validgen.n//validgen.batch_size,
                              epochs = epochs,
                              callbacks = [ CLR])



# CLR LR history plot
N = np.arange(0, len(CLR.history["lr"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, CLR.history["lr"])
plt.title("CLR")
plt.xlabel("Training Iterations")
plt.ylabel("LR")
plt.savefig('C:/Keras_Tutorial/Saved_Unet/UNET_Extra_Layers/Plots/Scheduler/custom_gen_latest/Plots/CLR_Plots/CLRP')

# F1 score plot
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