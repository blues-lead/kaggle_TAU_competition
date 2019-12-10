# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#os.listdir('/kaggle/input/vehicle/train/train')
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

dir1 = 'D:\\docs\\kaggle_sgn_competition\\vehicle\\train\\train'
dir2 = 'C:\\Users\\Anton\\Google Drive\\vehicle\\train\\train'
dir3 = '/kaggle/input/vehicle/train/train'
checkp_dir = '/kaggle/working/weights_best.hdf5'


base_clf = VGG16(include_top=False)


for layer in base_clf.layers[:11]: # 11
  layer.trainable = False
  
for layer in base_clf.layers:
    print(layer, layer.trainable)

input_tensor = base_clf.inputs[0]
output_tensor = base_clf.outputs[0]

output_tensor = GlobalAveragePooling2D()(output_tensor)

output_tensor = Dense(100,activation='relu')(output_tensor)
output_tensor = Dropout(0.5)(output_tensor)

output_tensor = Dense(100,activation='relu')(output_tensor)

output_tensor = Dense(17,activation='softmax')(output_tensor)

mdl = Model(inputs=[input_tensor], outputs=[output_tensor])

mdl.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])



print(mdl.summary())

#plot_model(mdl,to_file='model.png')

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
        dir2,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        dir2,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

checkpointer = ModelCheckpoint(checkp_dir, monitor='val_accuracy', save_best_only=True, period=1)
callback_list = [checkpointer]
#history=mdl.fit_generator(train_generator,
#                  steps_per_epoch=687,
#                  validation_steps=175,
#                  validation_data = validation_generator,
#                  epochs=20,
#                  callbacks=callback_list)

#mdl.save('/kaggle/fst_model.h5')