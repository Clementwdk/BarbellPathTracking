import cv2
import mediapipe as mp
import os
import numpy as np

path = 'ScreenShot'
picfiles = [f for f in os.listdir(path) if f.endswith('.PNG')]
filmfile = "videos/3.mp4"


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mpDrStyle = mp.solutions.drawing_styles


def bodyEstimationPics(pathPics):

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

def bodypose3d():
    BG_COLOR = (192, 192, 192) # gray
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192)  # gray
    with mpPose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread("ScreenShot/pic_43612.png")
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mpDraw.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=mpDrStyle.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mpDraw.plot_landmarks(
            results.pose_world_landmarks, mpPose.POSE_CONNECTIONS)



#bodyEstimationVideo(filmfile)

# print((picfiles))
# if not picfiles:
#     print("No Screen shot taken")
# else:
#     for x in picfiles:
#         bodyEstimationPics(path + '/' + x)

bodyEstimationPics("ScreenShot/pic_443612.png")