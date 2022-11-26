import depthai as dai
import lane_detection
import VESC_class as VESC

def main():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)

    xoutVideo.setStreamName("video")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)

    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    # Linking
    camRgb.video.link(xoutVideo.input)
    
    with dai.Device(pipeline) as device:

        # get video from camera
        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

        # Create VESC Object
        # carVESC = VESC('/dev/ttyACM0') 

        while True:
            # get frame from camera
            videoIn = video.get()
            orig_frame = videoIn.getCvFrame()

            # calculate steering and throttle based on lane_detection
            steering, throttle = lane_detection.main(orig_frame)

            print(steering, throttle)
            
            # Use VESC.run() to set steering and throttle
            # carVESC.run(steering,throttle)

if __name__ == '__main__':
    main()