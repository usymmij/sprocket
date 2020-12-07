import cv2
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

global model
global image
global label

#vid =cv2.VideoCapture(sys.argv[1])
vid = cv2.VideoCapture('data/a.mp4')
model = keras.models.load_model('model.h5')
succ, frame = vid.read()
image = np.load('images.npy')
label = np.load('labels.npy')


def a(i):     
    global model
    global image
    global label
    
    image = np.reshape(image, (-1,480,720,1))
    label = np.reshape(label, (-1,4))
    lb = label[i]
    frame = image[i]
    
    frame = cv2.resize(frame, (720, 480))
    fr = frame.copy()
    
    fr = fr.reshape((1, 480, 720, 1))
    out = model.predict(fr)
    print(out)
    out=out[0]
    print(lb)


    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = cv2.rectangle(frame, (out[0],out[1]), (out[2],out[3]), (255,100,150), 2)
    frame = cv2.rectangle(frame, (lb[0],lb[1]),   (lb[2],lb[3]),    (255,0,20),   2)
    
    cv2.imshow('a',frame)
    cv2.waitKey(0)
    return frame