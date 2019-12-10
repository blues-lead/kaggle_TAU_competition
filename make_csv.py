# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:17:21 2019

@author: Anton
"""

# print('{num:06d}'.format(num=i))

import os
from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

def main():
#    class_names = []
#    data_dir = 'C:\\Users\\Anton\\Google Drive\\vehicle\\train\\train'
#    for root,dirs,files in os.walk(data_dir):
#        class_names.append(root.split('\\')[-1])
#    class_names = sorted(class_names)
#
#    files = []
#    for i in range(7958):
#        path = 'C:\\Users\\Anton\\Google Drive\\vehicle\\test\\testset\\'
#        str = '{num:06d}.jpg'.format(num=i)
#        #print(path + str)
#        files.append(path + str)
        
    mdl = load_model('inceptionv3_Anton2_p50.h5')
#    predictions = []
#    for file in files:
#        img = plt.imread(file)
#        img = resize(img, (299,299))
#        img = img.astype(np.float32)
#        predict = mdl.predict(img[np.newaxis,...])[0]
#        predict = [float(i) for i in predict]
#        predictions.append(class_names[predict.index(np.max(predict))])
#    print(predictions)
#    
main()