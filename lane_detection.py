import cv2
import numpy as np
import controller_class as mC
        
def in_bounds(x1, x2, y1, y2, width, height): # function to make sure points in image

    # if you remove this function, the lines actually follow edge detection
    # but, this is used for when the algorithm doesn't detect any 
    # and is life saver otherwise no data

    offset = 20

    if x1 > width:
        x1 = width - offset
    elif x1 < 0:
        x1 = 0

    if x2 > width:
        x2 = width 
    elif x2 < 0:
        x2 = offset

    if y1 > height:
        y1 = height
    elif y1 < 0:
        y1 = 0
    
    if y2 > height:
        y2 = height
    elif y2 < 0:
        y2 = 0

    return x1, y1, x2, y2

def previous_weight(x1o, x2o, y1o, y2o, x1n, x2n, y1n, y2n, scale):
    # function to scale new points based on previous ones
    # higher scale values means more weight on older values

    x1 = int(scale*x1o + (1-scale)*x1n)
    x2 = int(scale*x2o + (1-scale)*x2n)
    y1 = int(scale*y1o + (1-scale)*y1n)
    y2 = int(scale*y2o + (1-scale)*y2n)

    return x1, x2, y1, y2

def find_canny(img,thresh_low,thresh_high): #function for implementing the canny

    # turns image to monocolor to make it easier to detect contrast in edges
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

    # blurs the image to emphasize color contrast
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)

    # detects edges on the blurred image
    img_canny = cv2.Canny(img_blur,thresh_low,thresh_high)
    return img_canny

def region_of_interest(image, width, height): #function for extracting region of interest

    # identify bounds in (x,y) format
    # currently targeting the lower half of the image
    bounds = np.array([[[0,height],[0,height/2],[width,height/2],[width,height]]],dtype=np.int32)

    # creates mask and returns specified region
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,bounds,[255,255,255])
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def draw_lines(img,lines,width,height): #function for drawing lines on black mask
    mask_lines=np.zeros_like(img)
    for points in lines:
        x1,y1,x2,y2 = points[0]

        # if point is past image size, reset to max or min location
        x1,y1,x2,y2 = in_bounds(x1,x2,y1,y2,width,height)

        # cv2.line(mask_lines,(x1,y1),(x2,y2),[0,0,255],2)

    return mask_lines

def get_coordinates(img,line_parameters): #functions for getting final coordinates
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 = img.shape[0]
    y2 = 0.6*img.shape[0]
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1,int(y1),x2,int(y2)]

def compute_average_lines(img,lines,width,height):

    left_lane_lines=[]
    right_lane_lines=[]
    left_weights=[]
    right_weights=[]
    for points in lines:
        x1,y1,x2,y2 = points[0]
        if x2==x1:
            continue     

        #implementing polyfit to identify slope and intercept
        parameters = np.polyfit((x1,x2),(y1,y2),1) 
        slope,intercept = parameters
        length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        if slope <0:
            left_lane_lines.append([slope,intercept])
            left_weights.append(length)         
        elif slope > 0:
            right_lane_lines.append([slope,intercept])
            right_weights.append(length)

    # Computing average slope and intercept
    left_average_line = np.average(left_lane_lines,axis=0)
    right_average_line = np.average(right_lane_lines,axis=0)
    

    # Computing weigthed sum
    if len(left_weights)>0:
        left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    if len(right_weights)>0:
        right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)

    # if left lane is not detected, create own left lane
    # bias the line towards the left
    try:
        left_fit_points = get_coordinates(img,left_average_line)
    except:
        left_fit_points = [0, height - 400, 200, height]
        # print("no left")

    # if right lane is not detected, create own right lane
    # bias the line towards the right
    try:
        right_fit_points = get_coordinates(img,right_average_line)
    except:
        right_fit_points = [width - 200, height, width, height - 400]
        # print("no right")

    return [[left_fit_points],[right_fit_points]] #returning the final coordinates

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

    # counters
    count1 = 0 # no lines counter
    count2 = 0 # no avg lines counter
    
    # initialize tuning parameters
    canny_lower = 60; canny_upper = 110
    hl_thresh = 30; hl_minLength = 20; hl_maxGap = 10
    scale = 0.1

    while True_Counter:
        if __name__ == '__main__':
            ret, orig_frame = video.read()
        else:
            orig_frame = video

        # make copy of image
        frame = np.copy(orig_frame)
        height = frame.shape[0]
        width  = frame.shape[1]

        # implement canny function and HL transform on ROI
        edges = find_canny(frame, canny_lower, canny_upper)
        lane_roi = region_of_interest(edges, width, height)
        lines = cv2.HoughLinesP(lane_roi, 1, np.pi/180, hl_thresh, hl_minLength, hl_maxGap)

        # wrapped in try and except because crashes if no line detected
        # not used but kept to log if a frame doesn't have any lines processed
        try:
            lane_lines_plotted = draw_lines(frame,lines,width,height)
        except:
            count1 += 1

        # wrapped in t/e because creahes if no average line detected
        try:
            # computes a left and right line based on all lines detected
            result_lines = compute_average_lines(frame,lines,width,height)
            final_lines_mask = draw_lines(frame,result_lines,width,height)

            if result_lines is not None:
                # plots line on final image

                line_counter = 0 # counter for number of lines

                for line in result_lines:
                    x1,y1,x2,y2 = line[0]
                    # if point is past image size, reset to max or min location
                    x1,y1,x2,y2 = in_bounds(x1,x2,y1,y2,width,height)

                    # adds previous weight to points after a while
                    if count1 > 10:
                        x1,y1,x2,y2 = previous_weight(x1o,x2o,y1o,y2o,x1,y1,x2,y2,scale)

                    # edge detection lines
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

                    # to be used to find average value
                    xSum = x1 + x2; ySum = y1 + y2;
                    line_counter += 1

                xAvg = int(xSum/line_counter); yAvg = int(ySum/line_counter)

                # gets angle, steering, and throttle value
                angle, steering, throttle = car.getControllerVal(xAvg,yAvg,width,height)
                x1s, x2s, y1s, y2s = car.steeringLine(angle, throttle, width, height)
                cv2.line(frame, (x1s, y1s), (x2s, y2s), (255, 255, 255), 10)    

                # set new points to previous points
                x1o = x1; x2o = x2; y1o = y1; y2o = y2
        except:
            # logs if a frame doesn't have an average line
            count2 += 1

            # if unable to get steering or throttle value, program crashes
            # forces car to go straight at low moving speed  
            steering = 0.5 
            throttle = 0.1

        cv2.imshow("frame", frame)

        # press q to break program
        # remove in final product
        if __name__ == '__main__':
            if cv2.waitKey(1) & 0xff ==ord('q'):
                print("Frames with no lines detected:", count1)
                print("Frames with no avg lines detected:", count2)
                break
        else:
            cv2.waitKey(1)
            True_Counter = False
            return steering, throttle

if __name__ == '__main__':
    main()