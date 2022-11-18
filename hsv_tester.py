import cv2
import numpy as np

image = cv2.imread('images/3473_cam_image_array_.jpg') #Reading the image file
lane_image = np.copy(image)
lane_image = cv2.resize(lane_image,(650,500))

frame = cv2.GaussianBlur(lane_image, (5, 5), 0)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([0,0,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(lane_image,lane_image, mask= mask)

edges = cv2.Canny(mask, 35,115)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 31,24,19)

cv2.imshow("frame", lane_image)
cv2.imshow("mask", mask)
cv2.imshow("res", res)
key = cv2.waitKey(0)