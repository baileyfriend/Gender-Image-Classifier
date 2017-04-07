#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:39:42 2017

@author: baileyfreund
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications

def save_bottleneck_features():
    img_height, img_width = 250, 250
    batch_size = 16
    num_training_samples = 3367 #these numbers will change when more data is added
    num_validation_samples = 1518
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
            
            
        datagen = ImageDataGenerator(rescale =1./255)
        
        model = applications.VGG16(include_top = False, weights = 'imagenet')
        
        generator = datagen.flow_from_directory(
                'data/train',
                target_size=(img_height, img_width),
                batch_size = batch_size,
                class_mode = None,
                shuffle = False
                )
        bottleneck_features_train = model.predict_generator(generator, num_training_samples)
        np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
        
        
        generator = datagen.flow_from_directory(
                'data/validation',
                target_size=(img_height, img_width),
                batch_size = batch_size,
                class_mode = None,
                shuffle = False
                )
        bottleneck_features_validation = model.predict_generator(generator, num_validation_samples)
        np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)



def train_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (num_training_samples/2) + [1] * (num_training_samples/2))

    validation_data = np.adlo(open('bottleneck_features_validation.npy'))
    train_labels = np.array([0] * (num_validation_samples/2) + [1] * ((num_validation_samples/2)))
    
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer = 'rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_data, train_labels,
              epochs=5,
              batch_size=batch_size,
              validation_data=(validation_data,validation_labels))
    
    model.save_weights('bottleneck_model.h5')


save_bottleneck_features()
train_model()