import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import wandb
from wandb.keras import WandbCallback
wandb.init(project="sprocket")

EPOCHS = 50
BATCHES = 1

model = keras.models.load_model('model.h5')

lb = np.load('labels.npy')
im = np.load('images.npy')

im = im.reshape((-1, 480, 720,1))
print(im.shape)

lb = lb.reshape(-1, 4)
print(lb.shape)

t = 0

while (t < 30):
    model.fit(im, lb, BATCHES, EPOCHS, shuffle=True, callbacks=[WandbCallback()])
    #model.save('/media/student/DARKGREEN32/modelSaves/'+str(t)+'.h5')
    model.save('./saves/'+str(t)+'.h5')
    t += 1
