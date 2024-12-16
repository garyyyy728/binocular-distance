"""Model management and switching interface"""
from ..detection.yolo_detector import YOLODetector
from ..config.model_config import SUPPORTED_MODELS
from ..utils.exceptions import DetectionError

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.supported_models = SUPPORTED_MODELS
        
    def list_models(self):
        """列出所有支援的模型"""
        print("\n可用的模型：")
        for model_type, models in self.supported_models.items():
            print(f"\n{model_type.upper()}:")
            for name, info in models.items():
                print(f"  - {name}: {info['description']}")
                
    def load_model(self, model_type, model_name):
        """載入指定的模型"""
        try:
            if model_type not in self.supported_models:
                raise DetectionError(f"不支援的模型類型: {model_type}")
                
            if model_name not in self.supported_models[model_type]:
                raise DetectionError(f"找不到模型: {model_name}")
                
            model_path = self.supported_models[model_type][model_name]['path']
            
            if model_type == 'yolo':
                self.current_model = YOLODetector(model_path)
            elif model_type == 'unet':
                # TODO: 實現 U-Net 模型載入
                raise NotImplementedError("U-Net 模型支援即將推出")
                
            self.current_model_name = f"{model_type}_{model_name}"
            print(f"\n成功載入模型: {model_name}")
            
        except Exception as e:
            raise DetectionError(f"載入模型失敗: {str(e)}")
            
    def get_current_model(self):
        """獲取當前使用的模型"""
        if self.current_model is None:
            raise DetectionError("尚未載入任何模型")
        return self.current_model