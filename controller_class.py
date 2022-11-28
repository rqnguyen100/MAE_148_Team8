import numpy as np

class motorController():
    def getControllerVal(self, xAvg, yAvg, width, height):
        # getter function to get steering and throttle
        angle = self.calcAngle(xAvg,yAvg,width,height)

        steering = self.calcSteering(angle)
        throttle = self.calcThrottle(angle)

        self.steering = steering
        self.throttle = throttle

        return angle, steering, throttle
    
    def steeringLine(self, angle, throttle, frameWidth, frameHeight):
        # implement visual aid to show direction on frame

        # ensure that first point is bottom-center of img
        x1 = int(frameWidth/2); y1 = int(frameHeight)

        # scale line based on throttle val
        hypo_max = np.sqrt(x1**2 + y1**2)
        hypo = (hypo_max * throttle)/1.5 

        x2 = int(x1 + hypo*np.cos(angle))
        y2 = int(y1 - hypo*np.sin(angle))

        return x1, x2, y1, y2   
        
    def calcAngle(self, xAvg, yAvg, width, height):
        # calculate angle in radians based on given points
        x1 = int(width/2); x = xAvg - x1
        y = height - yAvg

        angle = np.arctan2(y,x)

        return angle

    def calcSteering(self, angle):
        # calculate a steering value based on angle calculated

        # angle interval: [-pi, pi]
        # servo interval: [0, 1] -> not sure if 0 is left and right
        # will assume 0.5 maps to pi/2, 0.0 maps to 0, 1.0 maps to pi
        angle2steering_constant = 1/np.pi

        if angle > 0:
            steering = angle2steering_constant*angle
        else:
            # car will only accept [0, 1], so set to 0 if negative angle
            steering = 0

        return steering

    def calcThrottle(self, angle):
        # calculate throttle value 
        
        ## increase throttle based on angle
        # throttle interval: [0.1, 0.5]
        slope = (0.5-0.1)/(np.pi/2)
        throttle = slope*angle + 0.1

        return throttle