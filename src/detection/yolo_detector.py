"""YOLO detector implementation"""
from ultralytics import YOLO
import cv2
from .detector_base import DetectorBase
from ..config.detection_config import YOLO_CONFIG

class YOLODetector(DetectorBase):
    def __init__(self):
        self.model = YOLO(YOLO_CONFIG['model_path'])
        self.conf_threshold = YOLO_CONFIG['confidence_threshold']
        
    def detect(self, image):
        results = self.model(
            image,
            conf=self.conf_threshold
        )
        return results[0]
    
    def draw_detections(self, image, results):
        annotated_image = image.copy()
        
        for r in results.boxes.data:
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with class name and confidence
            label = f'{results.names[int(class_id)]} {score:.2f}'
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
        return annotated_image