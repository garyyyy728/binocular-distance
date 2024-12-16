"""Visualization utilities"""
import cv2
import numpy as np

def create_depth_visualization(disparity_normalized, color_image, detections):
    """Create visualization combining depth map and detections"""
    depth_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    visualization = np.hstack((color_image, depth_colormap))
    return visualization

def show_images(left_ir, right_ir, depth_vis, window_name='Depth Measurement'):
    """Display images in windows"""
    # Resize images for better visualization
    scale = 0.8
    left_ir = cv2.resize(left_ir, None, fx=scale, fy=scale)
    right_ir = cv2.resize(right_ir, None, fx=scale, fy=scale)
    depth_vis = cv2.resize(depth_vis, None, fx=scale, fy=scale)
    
    cv2.imshow('Left IR', left_ir)
    cv2.imshow('Right IR', right_ir)
    cv2.imshow(window_name, depth_vis)

def draw_text_info(image, text, position=(30, 30)):
    """Draw text information on image"""
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )