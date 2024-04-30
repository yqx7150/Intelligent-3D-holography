import pyrealsense2 as rs
import numpy as np
import cv2


class Depthcamera_ctrl():
    def __init__(self) -> None:
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.hole_filling = rs.hole_filling_filter()

    def get_depth_scale(self):
        return self.depth_scale

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_frame = self.hole_filling.process(depth_frame)  # !

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image


if __name__ == "__main__":
    D = Depthcamera_ctrl()

    while True:
        depth, image = D.get_frame()
        cv2.namedWindow('Align RGB', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Align RGB", 848, 480)
        cv2.namedWindow('Align depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Align depth", 848, 480);
        cv2.imshow('Align RGB', image) 
        # print(depth.max(), depth.min())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Align depth', depth_colormap)

        cv2.waitKey(10) 





    
