import random
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import mediapipe as mp
import os

video = cv2.VideoCapture('videos/3.mp4')
success, img = video.read()
boundBox = cv2.selectROI("Tracking, ", img, fromCenter=False, showCrosshair=True)
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(img, boundBox)
barbellPath = []
yhigh = []
xhigh = []
t_start =time.time()
sec = []
speed = []

#Track the bounding box return text on video if detected
def trackBox(img, boundBox):
    x, y, weidth, height = int(boundBox[0]), int(boundBox[1]), int(boundBox[2]), int(boundBox[3])
    cv2.rectangle(img,(x,y), ((x+weidth),(y+height)),(212,175,55),3,2)
    cv2.putText(img, "Barbell detected !! ", (75, 120), cv2.FONT_HERSHEY_COMPLEX, 0.7, (12,242,17), 2)
#Draw dot on the barbell path
def drawPath(img, barbellPath):
    for x in barbellPath:
        cv2.circle(img, x, 3, (0,0,255), -1)
#Calculate the speed of the barbell
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
    return distcm/timeElapsed
#take screenshot during the video to analyse the body position
def takeScreenshot():
    x= random.randrange(99999)
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("ScreenShot/pic_"+str(x)+".png", image)

#Create a graphic withe the move of the barbell
def graphMove(x,y):
    # #plotting
    plt.plot(2,1,1)
    plt.title("Athletes movement")
    plt.xlabel("Xsss")
    plt.ylabel("Ycsssss")
    plt.xticks(rotation=180)
    plt.plot(x, y, color ="red")
    plt.gca().invert_yaxis()
    plt.show()
#Create a graphicof the speed during the lifting
def graphSpeed(sec,speed):
    plt.title("Speed / sec")
    plt.xlabel("sec")
    plt.ylabel("M/S")
    plt.plot(sec, speed, 'blue')
    plt.show()

#Draw the pose landmarks on a pic
def bodyPose(pathPics):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(static_image_mode=True,
                       model_complexity=1,
                       smooth_landmarks=True,
                       enable_segmentation=False,
                       smooth_segmentation=True,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
    img = cv2.imread(pathPics)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            #print(id)
            # pixel value
            cx, cy = int(lm.x * w), int(lm.y * h)
            if True:
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

#Draw the body pos in all picture inside /Screenshot
def printBodypos():
    path = 'ScreenShot'
    picfiles = [f for f in os.listdir(path) if f.endswith('.png')]
    if not picfiles:
        print("No Screen shot taken")
    else:
        for x in picfiles:
            bodyPose(path + '/' + x)

#Delete the images inside /ScreenShot
def deleteScreenShot():
    dir_name = "ScreenShot"
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_name, item))

#Check if the file exist
if ( video.isOpened()== False):
    print("Error with the video file")


while True:

    timer = cv2.getTickCount()
    key = cv2.waitKey(1)
    success, img = video.read()
    success, boundBox = tracker.update(img)
    timeElapsed=time.time() - t_start

    barbellPath.append([int(boundBox[0]+(boundBox[2]/2)),int(boundBox[1]+boundBox[3]/2)])

    yhigh.append(boundBox[1])
    xhigh.append(boundBox[0])

    velocity = getVelocity(yhigh,int(boundBox[3]),timeElapsed)


    message = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    #Check the bounding box works
    if success == True:
        trackBox(img, boundBox)
    else:
        cv2.putText(img, "Barbell Undetected !! ", (75, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #Print informations on the video time, fps and Speed
    cv2.putText(img, "Time : {0:.2f} ".format(timeElapsed), (75, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (76,76,76), 2)
    cv2.putText(img, "FPS : {0:.2f} s".format(message), (75, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (76,76,76), 2)
    cv2.putText(img, "Speed : {0:.2f} m/s".format(velocity), (75, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (76,76,76), 2)

    #we start to collect the data when the athlete start to lift
    if velocity>0.5:
        sec.append(timeElapsed)
        speed.append(velocity)

    drawPath(img, barbellPath)

    #Press "s" to take a screenshot
    if cv2.waitKey(25) & key == ord('s'):
        cv2.waitKey(-1)
        takeScreenshot()
        print("ScreeShot taken")
    #Press "p" to pause
    if key == ord('p'):
        cv2.waitKey(-1)
    #Press "q" to quit
    if cv2.waitKey(25) & key == ord('q'):
        break

   #except if video is finish
    try:
        cv2.imshow('Tracking', img)
    except:
        print("video ended")

video.release()
cv2.destroyAllWindows()

graphSpeed(sec,speed)
graphMove(xhigh,yhigh)

printBodypos()
deleteScreenShot()
