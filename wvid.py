import cv2
import sys
import numpy as np

global boxes
global points
boxes = []
points = []
images = []

vid = cv2.VideoCapture(sys.argv[1])

def mouseEvent(event,x,y,flags,param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        pt = (x,y)
        points.append(pt)

succ = True
x = 0
while(succ):
    succ, frame = vid.read()

    if(x == 20):
        points = []

        cv2.namedWindow('a')
        cv2.setMouseCallback('a',mouseEvent)
        fr = cv2.resize(frame, (720, 480))
        fr= cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        cv2.imshow('a',fr)
        cv2.waitKey(0)
        print('adada')
        
        print(len(points))

        boxes.append(np.asarray(points))
        images.append(fr)
        x = 0
    x += 1
images = np.asarray(images)
boxes= np.asarray(boxes)
print(images.shape)
print(boxes.shape)
np.save("images.npy", images, True)
np.save("labels.npy", boxes, True)
imgs = np.load("images.npy")
lbls = np.load("labels.npy")

print(imgs)
print(lbls)