import cv2
import sys
import numpy as np

global boxes
global points
global images
boxes = []
points = []
images = []

global vid

def mouseEvent(event,x,y,flags,param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        pt = (x,y)
        points.append(pt)


def mmf(nom):
    global boxes
    global points
    global images
    boxes = []
    points = []
    images = []

    succ = True
    x = 0
    succ, frame = vid.read()
    while(succ):    
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
        succ, frame = vid.read()
        x += 1
    images = np.asarray(images)
    boxes= np.asarray(boxes)
    print(images.shape)
    print(boxes.shape)
    np.save(str(nom)+"images.npy", images, True)
    np.save(str(nom)+"labels.npy", boxes, True)  

if __name__ == "__main__":
    global vid
    for i in range(len(sys.argv)-1):
        vid = cv2.VideoCapture(sys.argv[i+1])
        print('video number: '+str(i))
        mmf(str(i))