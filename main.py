from imutils.video import videostream
from imutils.video import FPS
import time
import cv2
import imutils
import numpy as np

video = cv2.VideoCapture('Snatch.mp4')

# video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(str(video.get(3)))


success, img = video.read()
# img = imutils.resize(img, width=640, height=480)
boundBox = cv2.selectROI("Tracking, ", img, fromCenter=False, showCrosshair=True)
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(img, boundBox)
barbellPath = []

if ( video.isOpened()== False):
    print("Error with the video file")

def trackBox(img, boundBox):
    x, y, weidth, height = int(boundBox[0]), int(boundBox[1]), int(boundBox[2]), int(boundBox[3])
    cv2.rectangle(img,(x,y), ((x+weidth),(y+height)),(212,175,55),3,2)
    cv2.putText(img, "Barbell detected !! ", (75, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    #print(str(x )+"-----"+ str(y))
def drawPath(img, barbellPath):
    # coordinate = int(barbellPath[0]), int(barbellPath[1])
    for  x in barbellPath:
        print(x)
        cv2.circle(img, x, 7, (255,0,0), -1)
        #cv2.circle(img, [509,555], 7, (255,0,0),-1)

while True:
    fpss = FPS().start()
    timer = cv2.getTickCount()
    success, img = video.read()

    fpss.update()
    fpss.stop()

    success, boundBox = tracker.update(img)
    barbellPath.append([int(boundBox[0]),int(boundBox[1])])

    message = "{:.2f}".format(fpss.fps())
    if success == True:
        trackBox(img, boundBox)


        #cv2.circle(img, ([boundBox[0], boundBox[1]]), 7, (255, 0, 0), -1)

    else:
        cv2.putText(img, "Barbell Undetected !! ", (75, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, "FPS : "+str(message), (75, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    drawPath(img, barbellPath)

    cv2.imshow('Tracking', img)

    if (cv2.waitKey(25) & 0xFF == ord('q')):
        break


video.release()
cv2.destroyAllWindows()
print(barbellPath)
