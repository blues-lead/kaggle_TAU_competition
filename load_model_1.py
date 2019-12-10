# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:11:58 2019

@author: kondrate
"""

from tensorflow.keras.models import load_model

mdl = load_model('inceptionv3_Anton2_p50.h5')

import os
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

data_dir = 'C:\\Work\\sgndataset\\train'
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
mx = 0
for key in data:
    if len(data[key]) > mx:
        mx = len(data[key])

class_weights = {}
i = 0
for key in data:
    class_weights[i] = mx/len(data[key])
    i+=1
    
#print(class_weights)

rand_val = {}                
for key in data:
    rand_val[key] = random.choice(data[key])

for key in rand_val:
    print(key,rand_val[key])
    
for im in rand_val:
    img = plt.imread(rand_val[im])
    img = resize(img,(299,299))
    img = img.astype(np.float32)
    #img -= 128
    #predict = mdl.predict(img[np.newaxis,...])[0]
    print(mdl.predict(img[np.newaxis,...]))
#    predict = [float(i) for i in predict]
#    print("Ground truth:",im," | predicted:",class_names[predict.index(np.max(predict))])