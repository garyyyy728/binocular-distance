"""YOLO model trainer implementation"""
from ultralytics import YOLO
from pathlib import Path
from .training_config import TRAINING_CONFIG
from ..utils.exceptions import TrainingError

class YOLOTrainer:
    def __init__(self, config=None):
        """
        初始化 YOLO 訓練器
        
        Args:
            config (dict, optional): 訓練配置，默認使用 TRAINING_CONFIG
        """
        self.config = config or TRAINING_CONFIG
        self.model = None
        
    def setup_model(self):
        """設置和初始化 YOLO 模型"""
        try:
            # 載入預訓練模型或創建新模型
            self.model = YOLO(self.config['model_type'])
        except Exception as e:
            raise TrainingError(f"模型初始化失敗: {str(e)}")
            
    def train(self, data_yaml_path):
        """
        訓練 YOLO 模型
        
        Args:
            data_yaml_path (str): 數據配置文件路徑
        """
        if self.model is None:
            self.setup_model()
            
        try:
            # 開始訓練
            results = self.model.train(
                data=data_yaml_path,
                epochs=self.config['epochs'],
                imgsz=self.config['img_size'],
                batch=self.config['batch_size'],
                workers=self.config['workers'],
                optimizer=self.config['optimizer'],
                lr0=self.config['lr0'],
                weight_decay=self.config['weight_decay'],
                hsv_h=self.config['augmentation']['hsv_h'],
                hsv_s=self.config['augmentation']['hsv_s'],
                hsv_v=self.config['augmentation']['hsv_v'],
                degrees=self.config['augmentation']['degrees'],
                translate=self.config['augmentation']['translate'],
                scale=self.config['augmentation']['scale'],
                shear=self.config['augmentation']['shear'],
                flipud=self.config['augmentation']['flipud'],
                fliplr=self.config['augmentation']['fliplr'],
                mosaic=self.config['augmentation']['mosaic'],
                mixup=self.config['augmentation']['mixup']
            )
            return results
            
        except Exception as e:
            raise TrainingError(f"模型訓練失敗: {str(e)}")
            
    def validate(self, data_yaml_path):
        """
        驗證訓練後的模型
        
        Args:
            data_yaml_path (str): 數據配置文件路徑
        """
        if self.model is None:
            raise TrainingError("沒有可用的模型進行驗證")
            
        try:
            results = self.model.val(data=data_yaml_path)
            return results
        except Exception as e:
            raise TrainingError(f"模型驗證失敗: {str(e)}")