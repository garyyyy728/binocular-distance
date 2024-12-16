"""SGBM depth processor implementation"""
import cv2
import numpy as np
from .depth_processor_base import DepthProcessorBase
from ..config.depth_config import SGBM_CONFIG

class SGBMProcessor(DepthProcessorBase):
    def __init__(self):
        self.window_size = SGBM_CONFIG['window_size']
        self.min_disp = SGBM_CONFIG['min_disparity']
        self.num_disp = SGBM_CONFIG['num_disparities']
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=SGBM_CONFIG['disp12_max_diff'],
            uniquenessRatio=SGBM_CONFIG['uniqueness_ratio'],
            speckleWindowSize=SGBM_CONFIG['speckle_window_size'],
            speckleRange=SGBM_CONFIG['speckle_range']
        )
        
    def compute_depth(self, left_image, right_image):
        # Compute disparity
        disparity = self.stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(
            disparity, None,
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )
        
        return disparity, disparity_normalized