"""Model configuration and registry"""

# 支援的模型配置
SUPPORTED_MODELS = {
    'yolo': {
        'yolov8n': {
            'path': 'yolov8n.pt',
            'description': 'YOLOv8 Nano - 快速但精度較低'
        },
        'yolov8s': {
            'path': 'yolov8s.pt',
            'description': 'YOLOv8 Small - 平衡速度和精度'
        },
        'yolov8m': {
            'path': 'yolov8m.pt',
            'description': 'YOLOv8 Medium - 較高精度'
        },
        'yolov8l': {
            'path': 'yolov8l.pt',
            'description': 'YOLOv8 Large - 高精度'
        }
    },
    'unet': {
        'unet_small': {
            'path': 'unet_small.pt',
            'description': 'U-Net Small - 輕量級分割模型'
        },
        'unet_base': {
            'path': 'unet_base.pt',
            'description': 'U-Net Base - 標準分割模型'
        }
    }
}

# 追蹤器配置
TRACKER_CONFIG = {
    'max_disappeared': 30,  # 物體消失多少幀後停止追蹤
    'max_distance': 50,    # 最大匹配距離
    'min_confidence': 0.3  # 最小檢測置信度
}