# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:16:00 2021

@author: yishu
"""
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import albumentations as A
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, df, act_images, binary_mask, X_col,
                 batch_size, ftrain_data=True,
                 shuffle=True):
        self.count = 0
        self.X_col = X_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.act_images = act_images
        self.binary_mask = binary_mask
        self.df = df
        self.n = len(self.act_images)
        self.ftrain_data = ftrain_data
        self.transform = transform = A.Compose([A.VerticalFlip(p=0.8), A.RandomRotate90(p=0.8),A.HorizontalFlip(p=0.8),A.Transpose(p=0.8)])
                                                #A.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5)
   
    def __len__(self):
        return self.n // self.batch_size

    def __get_input(self, idN):
        idN = idN[0].split('=')[1].split('\"')[1]  # For getting the id as incoming idN is a tupple of format ('="11616"',)
        image_arr_act = self.act_images[idN]
        image_arr_mask = self.binary_mask[idN]
        return image_arr_act, image_arr_mask

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        ids_batch = batches[self.X_col['ids']]

        X_batch_act = []
        y0_batch = []
        for idN in zip(ids_batch):
            act, mask = self.__get_input(idN)
            if(self.ftrain_data):
                    transformed = self.transform(image=act, mask=mask)
                    thresh = np.expand_dims(cv2.threshold(np.array(transformed['mask'], dtype= np.uint8),128,255,cv2.THRESH_BINARY)[1], axis=2)
                    transformed_img = np.expand_dims(transformed['image'], axis=2)
                    X_batch_act.append(transformed_img/255)
                    y0_batch.append(thresh/255)
            else:
                act = np.expand_dims(act, axis=2)
                X_batch_act.append(act/255)
                thresh = np.expand_dims(cv2.threshold(mask,128,255,cv2.THRESH_BINARY)[1], axis=2)
                y0_batch.append(thresh/255)
        X_batch_act = np.asarray(X_batch_act)
        y0_batch = np.asarray(y0_batch)
        return X_batch_act, y0_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X_act, y = self.__get_data(batches)
        return X_act, y

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)



'''
batch_size =16

train_df = pd.read_csv("C:\\Keras_Tutorial\\ids.csv")

#df = df.sample(frac=1).reset_index(drop=True)
#msk = np.random.rand(len(df)) < 0.9
#train_df = df[msk]
#valid_df = df[~msk]

train_df = train_df.sample(frac=1).reset_index(drop=True)

act_images = np.load("C:/Keras_Tutorial/Casia/train_valid_combined/act_images.npy", allow_pickle=True).item()
binary_mask = np.load("C:/Keras_Tutorial/Casia/train_valid_combined/binary_mask.npy", allow_pickle=True).item()

#act_images = np.squeeze(act_images)
valid_df = train_df.sample(frac=1).reset_index(drop=True)
#valid_act_images = np.load("act_images.npy", allow_pickle=True).item()
#valid_binary_mask = np.load("binary_mask.npy", allow_pickle=True).item()


traingen = CustomDataGen(train_df, act_images, binary_mask, X_col={'ids':'ids'}, 
                         batch_size=batch_size, ftrain_data=True)
'''
'''validgen = CustomDataGen(valid_df, act_images, binary_mask,
        X_col={'ids':'ids'}, batch_size=batch_size, ftrain_data=False)'''
'''
#x,y = next(traingen)


#model.fit_generator(traingen, epochs=50, steps_per_epoch = train_df.shape[0]/batch_size, validation_data=validgen, validation_steps=valid_df.shape[0]/batch_size, callbacks=[checkpoint, CustomCallback(), reduce_lr, early_stop])
'''