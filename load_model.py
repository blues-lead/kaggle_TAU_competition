# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:36:44 2019

@author: Anton
"""

import os
import random

data_dir = 'C:\\Users\\Anton\\Google Drive\\vehicle\\train\\train'
data = {}


for root, dirs, files in os.walk(data_dir):
    for name in files:
        if name.endswith('.jpg'):
            cs=root.split('\\')[-1]
            if cs in data.keys():
                data[cs].append(root + os.sep + name)
            else:
                data[cs] = [root+os.sep + name]
                
class_names = sorted(data.keys())

rand_val = {}                
for key in data:
    rand_val[key] = random.choice(data[key])

for key in rand_val:
    print(key,rand_val[key])
        
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform


from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

#with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
model  = load_model('TrainingResult\\fst_model.h5')
#with CustomObjectScope({'GlorotUniform': glorot_uniform()}):    
#    mdl = load_model('TrainingResult\\fst_model.h5', custom_objects={
#        'adadelta': lambda **kwargs: hvd.DistributedOptimizer(keras.optimizers.Adadelta(**kwargs))
#    })
#
#dir1 = 'D:\\docs\\kaggle_sgn_competition\\vehicle\\train\\train'
#dir2 = 'C:\\Users\\Anton\\Google Drive\\vehicle\\train\\train'
#dir3 = '/kaggle/input/vehicle/train/train'
#checkp_dir = '/kaggle/working/weights_best.hdf5'
#
#
#base_clf = VGG16(include_top=False)
#
#
##for layer in base_clf.layers[:11]: # 11
##  layer.trainable = False
##  
##for layer in base_clf.layers:
##    print(layer, layer.trainable)
#
#input_tensor = base_clf.inputs[0]
#output_tensor = base_clf.outputs[0]
#
#output_tensor = GlobalAveragePooling2D()(output_tensor)
#
#output_tensor = Dense(512,activation='relu')(output_tensor)
#output_tensor = Dropout(0.5)(output_tensor)
#
#output_tensor = Dense(100,activation='relu')(output_tensor)
#
#output_tensor = Dense(17,activation='softmax')(output_tensor)
#
#mdl = Model(inputs=[input_tensor], outputs=[output_tensor])
#
#mdl.load_weights('TrainingResult\\fst_model.h5')
#
#mdl.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#
#for im in rand_val:
#    img = plt.imread(rand_val[im])
#    img = cv2.resize(img,(224,224))
#    img = img.astype(np.float32)
#    img -= 128
#    predict = int(np.max(mdl.predict(img[np.newaxis,...])[0]))
#    print("Ground truth:",im,"predicted:",predict)
                
    
