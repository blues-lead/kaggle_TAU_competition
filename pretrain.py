# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:12:42 2019

@author: Anton
"""
# JUST START IT TODAY
import os
from PIL import Image
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    directory1 = 'C:\\Users\\Anton\\Google Drive\\vehicle\\train\\train'
    direcotry2 = 'D:\\docs\\kaggle_sgn_competition\\vehicle\\train\\train'
    os.chdir(directory1)
    class_names = sorted(os.listdir('.'))
    print(class_names)
    
    base_model = MobileNet(input_shape=(224,224,3),include_top=False)
    input_tensor = base_model.inputs[0]
    output_tensor = base_model.outputs[0]

    output_tensor = GlobalAveragePooling2D()(output_tensor)

    model = Model(inputs = [input_tensor],outputs=[output_tensor])
    
    model.compile(loss="categorical_crossentropy",optimizer="sgd")
    i = 0
    X = []
    y = []
    for root, dirs, files in os.walk('.'):
        for name in files:
            if name.endswith('.jpg'):
                i+=1
                img = plt.imread(root + os.sep + name)
                
                img = cv2.resize(img,(224,224))
                
                img = img.astype(np.float32)
                
                img -= 128
                
                x = model.predict(img[np.newaxis,...])[0]
                
                X.append(x)
                #print("Img",i,"read")
                label = root.split('\\')[-1]
                y.append(class_names.index(label))
                #print("Image:",label, i, "processed")
    X = np.array(X)
    y = np.array(y)
    np.save('feature_matrix',X)
    np.save('gt_vector',y)


main()