"""Base detector interface"""
from abc import ABC, abstractmethod

class DetectorBase(ABC):
    @abstractmethod
    def detect(self, image):
        pass
    
    @abstractmethod
    def draw_detections(self, image, results):
        pass