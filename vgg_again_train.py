# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#os.listdir('/kaggle/input/vehicle/train/train')
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
#import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
data_dir = 'C:\\Work\\sgndataset\\train'
checkp_dir = 'P:\\PycharmProjects\\competition\\checkpoints\\vgg19_st1\\checkpoint_{epoch:02d}-{val_loss:.2f}.hdf5'

data = {}
#================================ClassWeights==================================
for root, dirs, files in os.walk(data_dir):
    for name in files:
        if name.endswith('.jpg'):
            cs=root.split('\\')[-1]
            if cs in data.keys():
                data[cs].append(root + os.sep + name)
            else:
                data[cs] = [root+os.sep + name]
                
#class_names = sorted(data.keys())
mx = 0
for key in data:
    if len(data[key]) > mx:
        mx = len(data[key])

class_weights = {}
i = 0
for key in data:
    class_weights[i] = mx/len(data[key])
    i+=1
#==============================================================================

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        horizontal_flip=True,
        validation_split=0.2)
                
train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))

input_tensor = base_model.inputs[0]
output_tensor = base_model.outputs[0]

#for layer in base_model.layers:
#    layer.trainable = False
#    print(layer.name, layer.trainable)

output_tensor = GlobalAveragePooling2D()(output_tensor)
output_tensor = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(output_tensor)
output_tensor = Dropout(0.5)(output_tensor)
output_tensor = Dense(17, activation='softmax')(output_tensor)

mdl = Model(inputs = [input_tensor], outputs= [output_tensor])

checkpointer = ModelCheckpoint(checkp_dir, monitor='val_loss', verbose=1, save_best_only=False, save_freq='epoch', save_weights_only=False)
callback_list = [checkpointer]

mdl.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 

history=mdl.fit_generator(train_generator,
                  steps_per_epoch=700,
                  validation_steps=175,
                  validation_data = validation_generator,
                  epochs=10,
                  callbacks=callback_list,
                  class_weight=class_weights)

mdl.save('models\\vgg19_st1.h5')

