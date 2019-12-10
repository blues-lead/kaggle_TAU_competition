from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
from tensorflow.keras.models import load_model
import numpy as np

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

checkp_dir = 'checkpoints\\inception_base_st2\\checkpoint_{epoch:02d}-{val_loss:.2f}.hdf5'
checkp_fine_dir = 'checkpoints\\inception_fine\\checkpoint_{epoch:02d}-{val_loss:.2f}.hdf5'
# #model_dir = '/kaggle/working/inceptionv3.h5'
data_dir = 'C:\\Work\\sgndataset\\train'

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
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[249:]:
    layer.trainable = True

input_tensor = base_model.inputs[0]
output_tensor = base_model.outputs[0]

output_tensor = GlobalAveragePooling2D()(output_tensor)

output_tensor = Dense(512, activation='relu')(output_tensor)
output_tensor = Dropout(0.5)(output_tensor)

output_tensor = Dense(512, activation='relu')(output_tensor)
output_tensor = Dense(17, activation='softmax')(output_tensor)

mdl = Model(inputs=[input_tensor], outputs=[output_tensor])

    
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
        target_size=(299,299),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(299,299),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

checkpointer = ModelCheckpoint(checkp_dir, monitor='val_loss', verbose=1, save_best_only=False, save_freq='epoch', save_weights_only=False)
callback_list = [checkpointer]
mdl = load_model('checkpoints\\inception_base_st2\\checkpoint_21-0.74.hdf5')
mdl.compile(optimizer=RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy']) 

#mdl=load_model('checkpoints\\inception_base\\checkpoint_10-0.96.hdf5')

history=mdl.fit_generator(train_generator,
                  steps_per_epoch=700,
                  validation_steps=175,
                  validation_data = validation_generator,
                  epochs=50,
                  callbacks=callback_list,
                  class_weight=class_weights)
mdl.save('models\\inceptionv3_Anton2_p50.h5')
#print('='*80)
#print('Finetuning started')
#print('='*80)
#
##mdl = load_model('models\\inceptionv3_Anton.h5')
#
#for layer in mdl.layers[:249]:
#        layer.trainable = False
#for layer in mdl.layers[249:]:
#        layer.trainable = True
#
#checkpointer = ModelCheckpoint(checkp_fine_dir, monitor='val_loss', verbose=1, save_best_only=False, save_freq='epoch', save_weights_only=False)
#callback_list = [checkpointer]
#
#mdl.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#
#history=mdl.fit_generator(train_generator,
#                  steps_per_epoch=700,
#                  validation_steps=175,
#                  validation_data = validation_generator,
#                  epochs=54,
#                  callbacks=callback_list,
#                  class_weight=class_weights,
#                  verbose=2)
#
#mdl.save('models\\inceptionv3_Anton_tuned.h5')
#np.save('v3_hist.hst', history)