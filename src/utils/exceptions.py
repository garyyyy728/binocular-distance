"""Custom exceptions for the project"""

class CameraError(Exception):
    """Exception raised for camera-related errors"""
    pass

class DepthProcessingError(Exception):
    """Exception raised for depth processing errors"""
    pass

class DetectionError(Exception):
    """Exception raised for object detection errors"""
    pass

class DataPreparationError(Exception):
    """Exception raised for data preparation errors"""
    pass

class TrainingError(Exception):
    """Exception raised for model training errors"""
    pass