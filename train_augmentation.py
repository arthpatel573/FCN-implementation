import keras.models as models
from keras.layers import Merge, Input, ELU, concatenate, Add 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from helper import one_hot_it
from fcn8 import FCN32
from fcn32 import FCN8

import os
import cv2
import numpy as np

#FCN-8 model
print("FCN - 8")
fcn8 = FCN8(352,352,3,2)

#For FCN-32 model
#print("FCN - 32")
#fcn32 = FCN32(352,352,3,2)

###########################################################################################################
print("Loading Sky Images")

#The below folder contains images of 30 different types of sky shades.
#The images at this path are used for data augmentation. 
skyimagespath = 'Sky_Images'
skyimages = []
for filename in os.listdir(skyimagespath):
    skyimage = cv2.imread(os.path.join(skyimagespath,filename))
    skyimage = cv2.resize(skyimage,(352,352))
    skyimages.append(skyimage)
sky_data = np.array(skyimages)

###########################################################################################################

print("Loading Training data")

traindata = []
trainlabel = []
#path to training images dataset
datapath = 'CamVid/train'
#path to annotations of training images
labelpath = 'CamVid/trainannot'
skindex = 0
for filename in os.listdir(datapath):
    
    lbl = cv2.imread(os.path.join(labelpath,filename))
    lbl = cv2.resize(lbl,(352,352))
    lbl = one_hot_it(lbl)
    trainlabel.append(lbl)

    #performing one-hot encoding
    img = cv2.imread(os.path.join(datapath,filename))
    img = cv2.resize(img,(352,352))
    traindata.append(img.copy())

    #Performing data augmentation
    #replacing pixels labeled as sky with the pixels of images of different sky shades 
    replace = np.where(np.all(lbl == (1,0), axis = -1))
    skindex = (skindex+1)%sky_data.shape[0]
    trainlabel.append(lbl)
    img[replace] = sky_data[skindex][replace]
    traindata.append(img.copy())

    """
    for images in range(0,sky_data.shape[0]):
        trainlabel.append(lbl)
        img[replace] = sky_data[images][replace]
        traindata.append(img)    
    """

train_data = np.array(traindata)
train_label = np.array(trainlabel)

train_label = np.reshape(train_label,(367*2,352*352,2))

#np.save('data/train_data.npy',train_data)
#np.save('data/train_label.npy',train_label)

###########################################################################################################
print("Loading Validation data")

valdata = []
vallabel = []
#Path to validation images dataset
datapath = 'CamVid/val'
#Path to annotations for validation images
labelpath = 'CamVid/valannot'
skindex = 0
for filename in os.listdir(datapath):
    lbl = cv2.imread(os.path.join(labelpath,filename))
    lbl = cv2.resize(lbl,(352,352))

    #performing one hot encoding
    lbl = one_hot_it(lbl)
    vallabel.append(lbl)

    img = cv2.imread(os.path.join(datapath,filename))
    img = cv2.resize(img,(352,352))
    valdata.append(img.copy())

    #Performing data augmentation
    #replacing pixels labeled as sky with the pixels of images of different sky shades 
    replace = np.where(np.all(lbl == (1,0), axis = -1))
    skindex = (skindex+1)%sky_data.shape[0]
    vallabel.append(lbl)
    img[replace] = sky_data[skindex][replace]
    valdata.append(img.copy())
    
    """
    replace = np.where(np.all(lbl == (1,0), axis = -1))
    for images in range(0,sky_data.shape[0]):
        vallabel.append(lbl)
        img[replace] = sky_data[images][replace]
        valdata.append(img)    
    """

val_data = np.array(valdata)
val_label = np.array(vallabel)

print(val_label.shape)

val_label = np.reshape(val_label,(124*2,352*352,2))

#np.save('data/val_data.npy',val_data)
#np.save('data/val_label.npy',val_label)


# load the model:

#checkpoint
filepath="model-checkpoint-FCN.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

nb_epoch = 60
batch_size = 12


#fcn8.load_weights('model-checkpoint.hdf5')
history = fcn8.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1 , validation_data=(val_data, val_label), shuffle=True)                   


# This save the trained model weights to this file with number of epochs
fcn8.save_weights('weights/model_weight_{}.h5'.format(nb_epoch))
fcn8.save('model.h5')