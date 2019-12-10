from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

#checkp_dir = '/kaggle/working/checkpoint.cpkt'
#model_dir = '/kaggle/working/inceptionv3.h5'
#data_dir = '/kaggle/input/vehicle/train/train'

checkp_dir = '/kaggle/working/checkpoint.cpkt'
model_dir = '/kaggle/working/inceptionv3.h5'
data_dir = '/kaggle/input/vehicle/train/train'



for layer in base_model.layers:
    layer.trainable = False

input_tensor = base_model.inputs[0]
output_tensor = base_model.outputs[0]

output_tensor = GlobalAveragePooling2D()(output_tensor)

output_tensor = Dense(512, activation='relu')(output_tensor)
output_tensor = Dropout(0.5)(output_tensor)

output_tensor = Dense(512, activation='relu')(output_tensor)
output_tensor = Dense(17, activation='softmax')(output_tensor)

mdl = Model(inputs=[input_tensor], outputs=[output_tensor])

#for layer in mdl.layers:
#    print(layer, layer.trainable)
    
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

checkpointer = ModelCheckpoint(checkp_dir, monitor='val_loss', verbose=1, save_best_only=True, save_freq='epoch', save_weights_only=True)
callback_list = [checkpointer]
mdl.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
history=mdl.fit_generator(train_generator,
                  steps_per_epoch=687,
                  validation_steps=175,
                  validation_data = validation_generator,
                  epochs=20,
                  callbacks=callback_list)
mdl.save(model_dir)