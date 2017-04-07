#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:27:05 2017

All images are of size 250x250

@author: baileyfreund
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

img_height, img_width = 250, 250

num_training_samples = 3367 #these numbers will change when more data is added
num_validation_samples = 1518

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
#Creating structure of our model
model.add(Conv2D(32, (3,3), input_shape = input_shape)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2), dim_ordering="th"))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten()) #converts 3D feature map to 1D feature vector
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', #doing binary classification of gender - 1 = female, 0 = male
              optimizer='rmsprop',
              metrics=['accuracy']
              )


batch_size = 16

#configuring augmentations for training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip = True
        )

#Same but with testing
test_datagen = ImageDataGenerator(rescale = 1./255)

# This generator will read pictors found in 'data/train' and indefinitely
# Batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(250,250),
        batch_size=batch_size,
        class_mode='binary'
        )

#same thing but for validation
validation_generator = train_datagen.flow_from_directory(
        'data/validation',
        target_size=(250,250),
        batch_size=batch_size,
        class_mode='binary'
        )

model.fit_generator(
        train_generator,
        steps_per_epoch=num_training_samples // batch_size,
        epochs = 25,
        validation_data = validation_generator,
        validation_steps = num_validation_samples // batch_size
        )

model.save_weights('first_try.h5')
