import cv2
import numpy as np
import controller_class as mC
from lane_detection import region_of_interest

def single_contour(frame, thresh):
        # calculate moments of binary image
        M = cv2.moments(thresh)
 
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # divide by 0 error
            cX, cY = 0, 0

        # put text and highlight the center
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return cX, cY

def multi_contour(frame, thresh):
    # find contours in the binary image
    contours, rest = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # find each centroid 
    xAvg = 0; yAvg = 0 # to be used to find average x and y value
    zeroCounter = 0

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # divide by 0 error
            cX, cY = 0, 0
            zeroCounter += 1

        # find sum of contours
        xAvg += cX; yAvg += cY

        # add indicator onto image
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(frame, "yellow", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # find controller inputs
    xAvg /= (len(contours) - zeroCounter)
    yAvg /= (len(contours) - zeroCounter)

    return xAvg, yAvg

def main(video=cv2.VideoCapture("lap2.mp4")):
    # lap1.mp4 is unedited full track
    # lap2.mp4 is right turn tester
    # lap3.mp4 is edited full track
    # lap4.mp4 is straight tester
    # lap5.mp4 is left turn tester

    # initialize controller class
    car = mC.motorController()

    # needed to run file as main and as import
    True_Counter = True

    while True_Counter:
        if __name__ == '__main__':
            ret, orig_frame = video.read()
        else:
            orig_frame = video

        #Reading the image file
        frame_copy = np.copy(orig_frame)
        height = frame_copy.shape[0]
        width  = frame_copy.shape[1]
        frame = region_of_interest(frame_copy,width,height)

        # HSV detection for yellow
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 50, 10], dtype="uint8")
        upper = np.array([50, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        # convert image to grayscale image
        gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image,127,255,0)

        ## find controller inputs for one contour
        xAvg, yAvg = single_contour(frame, thresh)

        ## find controller inputs for multiple contours
        # xAvg, yAvg = multi_contour(frame, thresh)

        # find controller outputs
        angle, steering, throttle = car.getControllerVal(xAvg,yAvg,width,height)
        x1s, x2s, y1s, y2s = car.steeringLine(angle, throttle, width, height)
        cv2.line(frame, (x1s, y1s), (x2s, y2s), (255, 255, 255), 10)    

        # display the image
        cv2.imshow("Image", frame)

        # press q to break program
        # remove in final product
        if __name__ == '__main__':
            if cv2.waitKey(1) & 0xff ==ord('q'):
                break
        else:
            cv2.waitKey(1)
            True_Counter = False
            return steering, throttle

if __name__ == '__main__':
    main()