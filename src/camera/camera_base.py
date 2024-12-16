"""Base camera interface"""
from abc import ABC, abstractmethod

class CameraBase(ABC):
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def get_frames(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def is_opened(self):
        pass