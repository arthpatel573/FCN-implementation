import keras.models as models
from keras.layers import Merge, Input, ELU, concatenate, Add 
from helper import *
from keras.optimizers import Adam
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import os
import cv2
import numpy as np

def FCN32(img_rows,img_cols,num_channels,num_labels):

    inputs = Input((img_rows, img_cols, num_channels))

    # Block 1
    x = Conv2D(8, (1, 1), activation='relu', padding='same')(inputs)
    x = Conv2D(8, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2,padding ='same')(x)
    f1 = x

    # Block 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2,padding ='same')(x)
    f2 = x

    # Block 3
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2,padding ='same')(x)
    f3 = x

    # Block 4
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2,padding ='same')(x)
    f4 = x

    # Block 5
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    f5 = x


    o = f5
    o = Conv2D(64, (7, 7), activation='relu', padding='same')(o)
    o = Dropout(0.5)(o)
    o = Conv2D(64,(1, 1), activation='relu' , padding='same')(o)
    o = Dropout(0.5)(o)


    o = Conv2D(num_labels, (1, 1) , activation='relu')(o)
    soft = Conv2DTranspose(num_labels, kernel_size=(32,32), strides=32, activation='softmax')(o)
    
    soft = Reshape((352*352,num_labels),input_shape=(352,352,num_labels))(soft)
    
    model = models.Model(inputs=inputs,output=soft )
    model.summary()
    model.compile(optimizer=Adam(lr=1e-3),loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
    return model