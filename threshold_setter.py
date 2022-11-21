import cv2 as cv2 #importing the library
import numpy as np

def lowerCallback(val):
    global lower_thresh
    lower_thresh = val
    print('lower', lower_thresh)

def upperCallback(val):
    global upper_thresh
    upper_thresh = val
    print('upper', upper_thresh)

def threshCallback(val):
    global thresh
    thresh = val
    print('threshold', thresh)

def minCallback(val):
    global minLine
    minLine = val
    print('minLineLength', minLine)

def maxCallback(val):
    global maxGap
    maxGap = val
    print('maxLineGap', maxGap)

def find_canny(img,thresh_low,thresh_high): #function for implementing the canny
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur,thresh_low,thresh_high)
    return img_canny

def region_of_interest(image): #function for extracting region of interest
    #bounds in (x,y) format
    bounds = np.array([[[0,698],[0,250],[650,250],[650,698]]],dtype=np.int32)
    
    # bounds = np.array([[[0,image.shape[0]],[0,image.shape[0]/2],[900,image.shape[0]/2],[900,image.shape[0]]]],dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,bounds,[255,255,255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def draw_lines(img,lines): #function for drawing lines on black mask
    mask_lines=np.zeros_like(img)
    for points in lines:
        x1,y1,x2,y2 = points[0]
        cv2.line(mask_lines,(x1,y1),(x2,y2),[0,0,255],2)

    return mask_lines

## Initialization
image = cv2.imread('images_2/669_cam_image_array_.jpg') #Reading the image file
lane_image = np.copy(image)
lane_image = cv2.resize(lane_image,(650,500))
height = lane_image.shape[0]; width  = lane_image.shape[1]

## Trackbars
lower_thresh = 50; upper_thresh = 150
thresh=30; minLine=5; maxGap = 100
# cv2.namedWindow('My Trackbars')
# cv2.resizeWindow('My Trackbars', 400, 100)
# cv2.createTrackbar('Canny Lower','My Trackbars',0,250, lowerCallback)
# cv2.createTrackbar('Canny Upper','My Trackbars',0,250, upperCallback)
# cv2.createTrackbar('HL Thresh','My Trackbars',0,100, threshCallback)
# cv2.createTrackbar('HL MinLineLength','My Trackbars',0,100, minCallback)
# cv2.createTrackbar('HL MaxLineGap','My Trackbars',0,100, maxCallback)

# Range for upper range
hsv = cv2.cvtColor(lane_image,cv2.COLOR_BGR2HSV)
lower = np.array([22, 93, 0])
upper = np.array([45, 255, 255])
mask_yellow = cv2.inRange(hsv, lower, upper)
yellow_output = cv2.bitwise_and(lane_image, lane_image, mask=mask_yellow)

while True:
    ## set canny threshold values
    lane_canny = find_canny(yellow_output,lower_thresh,upper_thresh)
    lane_roi = region_of_interest(lane_canny)
    # cv2.imshow('canny frame', lane_roi)

    ## set HL threshold values
    lane_lines = cv2.HoughLinesP(lane_roi,rho=1,theta=np.pi/180,threshold=thresh,minLineLength=minLine,maxLineGap=maxGap)
    lane_lines_plotted = draw_lines(lane_image,lane_lines)
    cv2.imshow('track', lane_lines_plotted)
    # cv2.moveWindow('track', 200, 0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

cv2.destroyAllWindows()