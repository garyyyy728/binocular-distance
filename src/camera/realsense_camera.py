"""Intel RealSense D435 camera implementation"""
import pyrealsense2 as rs
import numpy as np
from .camera_base import CameraBase
from ..config.camera_config import CAMERA_CONFIG
from ..utils.exceptions import CameraError

class RealSenseCamera(CameraBase):
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._is_running = False
        self._configure_streams()
        
    def _configure_streams(self):
        try:
            # Configure streams
            self.config.enable_stream(
                rs.stream.infrared, 1,
                CAMERA_CONFIG['width'],
                CAMERA_CONFIG['height'],
                rs.format.y8,
                CAMERA_CONFIG['fps']
            )
            self.config.enable_stream(
                rs.stream.infrared, 2,
                CAMERA_CONFIG['width'],
                CAMERA_CONFIG['height'],
                rs.format.y8,
                CAMERA_CONFIG['fps']
            )
            self.config.enable_stream(
                rs.stream.color,
                CAMERA_CONFIG['width'],
                CAMERA_CONFIG['height'],
                rs.format.bgr8,
                CAMERA_CONFIG['fps']
            )
        except Exception as e:
            raise CameraError(f"Failed to configure RealSense streams: {str(e)}")
        
    def start(self):
        try:
            self.pipeline.start(self.config)
            self._is_running = True
        except Exception as e:
            raise CameraError(f"Failed to start RealSense camera: {str(e)}")
        
    def get_frames(self):
        if not self._is_running:
            raise CameraError("Camera is not running")
            
        try:
            frames = self.pipeline.wait_for_frames()
            left_ir_frame = frames.get_infrared_frame(1)
            right_ir_frame = frames.get_infrared_frame(2)
            color_frame = frames.get_color_frame()
            
            if not left_ir_frame or not right_ir_frame or not color_frame:
                raise CameraError("Failed to get valid frames")
            
            left_image = np.asanyarray(left_ir_frame.get_data())
            right_image = np.asanyarray(right_ir_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return left_image, right_image, color_image
            
        except Exception as e:
            raise CameraError(f"Error getting frames: {str(e)}")
    
    def stop(self):
        if self._is_running:
            try:
                self.pipeline.stop()
                self._is_running = False
            except Exception as e:
                raise CameraError(f"Failed to stop camera: {str(e)}")
    
    def is_opened(self):
        return self._is_running