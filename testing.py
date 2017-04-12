#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:04:53 2017

@author: baileyfreund
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
import h5py
import numpy as np
from keras.models import model_from_json
import cv2

# load json and create model
#json_file = open('model.json', 'r')
json_file = open('model_tanh.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("weights.best.hdf5")
#loaded_model.load_weights("weights.h5")
print("Loaded model from disk")


bailey = cv2.imread("bailey.jpg")
#bailey = np.random.randint(0,10,(250,250,3))
bailey = np.expand_dims(bailey, axis=0)
prediction_b = model.predict(bailey)

ella = cv2.imread("ella.jpg")
ella_t = np.expand_dims(ella, axis=0)
prediction_e = model.predict(ella_t)

katie = cv2.imread("katie.jpg")
k_tensor = np.expand_dims(katie, axis=0)
prediction_k = model.predict(k_tensor)

n = cv2.imread("neon.jpg")
n_tensor = np.expand_dims(n, axis=0)
prediction_n = model.predict(n_tensor)

dude = cv2.imread("Christian_Longo_0001.jpg")
#dude = np.random.randint(0,10,(250,250,3))
dude = np.expand_dims(dude, axis=0)
prediction_d = model.predict(dude)
print("Dude : %s" % prediction_d)

f1 = cv2.imread("f1.jpg")
#dude = np.random.randint(0,10,(250,250,3))
f1 = np.expand_dims(f1, axis=0)
prediction_f1 = model.predict_classes(f1)
print("F1 : %s" % prediction_f1)

f2 = cv2.imread("f2.jpg")
#dude = np.random.randint(0,10,(250,250,3))
f2 = np.expand_dims(f2, axis=0)
prediction_f2 = model.predict(f2)
print("F2 : %s" % prediction_f2)


f3 = cv2.imread("f3.jpg")
f3 = np.expand_dims(f3, axis=0)
prediction_f3 = model.predict(f3)
print("F3 : %s" % prediction_f3)

eva = cv2.imread("eva.jpg")
eva = np.expand_dims(eva, axis=0)
prediction_eva = model.predict(eva)
print("Eva Mendes : %s" % prediction_eva)

ct = cv2.imread("CarrotTop.jpg")
#dude = np.random.randint(0,10,(250,250,3))
ct = np.expand_dims(ct, axis=0)
prediction_ct = model.predict(ct)
print("Carrot Top : %s" % model.predict(ct))

print("Bailey : %s"  % prediction_b)
print("Ella : %s"  % prediction_e)
print("Katie : %s"  % prediction_k)
print("Neon : %s"  % prediction_n)


