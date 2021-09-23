import cv2
import mediapipe as mp
import os


path = 'ScreenShot'
picfiles = [f for f in os.listdir(path) if f.endswith('.PNG')]

def bodyEstimation(pathPics):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(static_image_mode=True,
                       model_complexity=1,
                       smooth_landmarks=True,
                       enable_segmentation=False,
                       smooth_segmentation=True,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
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

print((picfiles))
if not picfiles:
    print("No Screen shot taken")
else:
    for x in picfiles:
        bodyEstimation(path + '/' + x)