"""YOLO training configuration"""
from pathlib import Path

TRAINING_CONFIG = {
    # 數據集配置
    'dataset_path': 'dataset',
    'train_ratio': 0.8,
    
    # 訓練參數
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'workers': 8,
    
    # 模型配置
    'model_type': 'yolov8n.pt',  # 可選: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    'pretrained': True,
    
    # 優化器配置
    'optimizer': 'Adam',
    'lr0': 0.01,
    'weight_decay': 0.0005,
    
    # 數據增強
    'augmentation': {
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 0.0,   # Rotation
        'translate': 0.1, # Translation
        'scale': 0.5,    # Scale
        'shear': 0.0,    # Shear
        'flipud': 0.0,   # Flip up-down
        'fliplr': 0.5,   # Flip left-right
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.0     # Mixup augmentation
    }
}