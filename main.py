import time
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture('Snatch.mp4')

#Settings  to modify the window size -> Fail
# video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# print(str(video.get(3)))




success, img = video.read()
#img = imutils.resize(img, width=640, height=480)
boundBox = cv2.selectROI("Tracking, ", img, fromCenter=False, showCrosshair=True)
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(img, boundBox)
barbellPath = []
yhigh = []
xhigh = []
t_start =time.time()
sec = []
speed = []


def trackBox(img, boundBox):
    x, y, weidth, height = int(boundBox[0]), int(boundBox[1]), int(boundBox[2]), int(boundBox[3])
    cv2.rectangle(img,(x,y), ((x+weidth),(y+height)),(212,175,55),3,2)
    cv2.putText(img, "Barbell detected !! ", (75, 120), cv2.FONT_HERSHEY_COMPLEX, 0.7, (12,242,17), 2)
def drawPath(img, barbellPath):
    for x in barbellPath:
        cv2.circle(img, x, 3, (255,0,0), -1)

def getVelocity(yhigh, heightBb, timeElapsed):
    yy=[]
    ymax = int(max(yhigh))
    for x in yhigh:
        if x>0:
            yy.append(x)
    ymin = int(yy[-1])
    try:
        distcm = ((ymax-ymin)/heightBb)*45
    except:
        distcm = 0
    # print(ymin, ymax, heightBb)
    # print("dist"+str(distcm))
    return distcm/timeElapsed

if ( video.isOpened()== False):
    print("Error with the video file")

while True:
    timer = cv2.getTickCount()
    key = cv2.waitKey(1)
    success, img = video.read()
    success, boundBox = tracker.update(img)
    timeElapsed=time.time() - t_start
    cv2.line(img, (100,8), (400,8), (212,175,55), 10)
    cv2.line(img, (100,541), (400,541), (24,175,55), 10)

    barbellPath.append([int(boundBox[0]+(boundBox[2]/2)),int(boundBox[1]+boundBox[3]/2)])

    yhigh.append(boundBox[1])
    xhigh.append(boundBox[0])

    velocity = getVelocity(yhigh,int(boundBox[3]),timeElapsed)


    message = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    if success == True:
        trackBox(img, boundBox)
    else:
        cv2.putText(img, "Barbell Undetected !! ", (75, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(img, "Time : {0:.2f} ".format(timeElapsed), (75, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (76,76,76), 2)
    cv2.putText(img, "FPS : {0:.2f} s".format(message), (75, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (76,76,76), 2)
    cv2.putText(img, "Speed : {0:.2f} m/s".format(velocity), (75, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (76,76,76), 2)

    if velocity>0.5:
        sec.append(timeElapsed)
        speed.append(velocity)

    drawPath(img, barbellPath)
    if key == ord('p'):
        time.sleep(2)
        cv2.waitKey(-1)

    if cv2.waitKey(25) & key == ord('q'):
        break
    try:
        cv2.imshow('Tracking', img)
    except:
        print("video ended")

video.release()
cv2.destroyAllWindows()

# #plotting
plt.plot(2,1,1)
plt.title("Athletes movement")
plt.xlabel("Xsss")
plt.ylabel("Ycsssss")
plt.xticks(rotation=180)
plt.plot(xhigh, yhigh, color ="red")
plt.show()

#plot speed
plt.subplot(2, 1, 2)
plt.title("Speed / sec")
plt.xlabel("sec")
plt.ylabel("M/S")
plt.plot(sec, speed, 'blue')

plt.show()
