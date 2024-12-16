import cv2
import numpy as np

class StereoDepth:
    def __init__(self):
        # SGBM Parameters
        self.window_size = 3
        self.min_disp = 0
        self.num_disp = 112
        
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
    def compute_depth(self, left_image, right_image):
        # Compute disparity
        disparity = self.stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return disparity, disparity_normalized