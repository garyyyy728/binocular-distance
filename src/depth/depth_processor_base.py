"""Base depth processor interface"""
from abc import ABC, abstractmethod

class DepthProcessorBase(ABC):
    @abstractmethod
    def compute_depth(self, left_image, right_image):
        pass