from imutils.video import videostream
from imutils.video import FPS
import time
import cv2
import imutils
import numpy as np

video = cv2.VideoCapture('Snatch_Trim.mp4')
success, img = video.read()
# img = cv2.resize(img1, (960, 540))
boundBox = cv2.selectROI("Tracking, ", img, fromCenter=False, showCrosshair=True)
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(img, boundBox)

if ( video.isOpened()== False):
    print("Error with the video file")

def trackBox(img, boundBox):
    x, y, weidth, height = int(boundBox[0]), int(boundBox[1]), int(boundBox[2]), int(boundBox[3])
    cv2.rectangle(img,(x,y), ((x+weidth),(y+height)),(212,175,55),3,1)
    cv2.putText(img, "Barbell detected !! ", (75, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)


while True:
    timer = cv2.getTickCount()
    success, img = video.read()

    success, boundBox = tracker.update(img)
    print(boundBox)

    if success == True:
        trackBox(img, boundBox)
    else:
        cv2.putText(img, "Barbell Undetected !! ", (75, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, "FPS : " + str(fps), (75, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Tracking', img)


    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break


video.release()
cv2.destroyAllWindows()