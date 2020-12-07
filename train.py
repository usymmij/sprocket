import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

BATCHES=1
EPOCHS=10
input_shape = (480, 720, 1)

lb = np.load('labels.npy')
im = np.load('images.npy')
print(im.shape)
'''
def inception_module(filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = layers.Conv2D(filters_1x
    keras.layers.Dropout(0.1),
    conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)
   
    conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)

    pool_proj = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)

    output = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

model = keras.Sequential([
    layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), input_shape=input_shape, activation='relu', 
    name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init),
    layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2'),
    layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1'),
    layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1'),
    layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2'),
    inception_module(filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a'),
    ])
'''
model = keras.models.Sequential([
    keras.layers.Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.05),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation="relu"),
    keras.layers.Dense(4096, activation="relu"),
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.BatchNormalization()
])

model.summary()

im = im.reshape((-1, 480, 720,1))
print(im.shape)

lb = lb.reshape(-1, 4)
print(lb.shape)

model.compile(optimizer="adam",
    loss="mae",
    metrics=["mae"])

model.fit(im, lb, BATCHES, EPOCHS, shuffle=True)
model.save("model.h5")


