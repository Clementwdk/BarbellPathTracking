import cv2
import numpy as np
import time

img = cv2.imread("videos/image/snatch.JPG")

while True:

    cv2.imshow("Image", img)
    cv2.waitKey(1)