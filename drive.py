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

        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

        # Create VESC Object

        while True:
            videoIn = video.get()

            orig_frame = videoIn.getCvFrame()

            # Get steering and throttle from lane_detection

            # If necessary, use previous value and add weight
            
            # Use VESC.run() to set steering and throttle

if __name__ == '__main__':
    main()