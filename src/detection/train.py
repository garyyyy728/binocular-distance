"""YOLO training script"""
import argparse
from pathlib import Path
from data_preparation import DataPreparation
from model_trainer import YOLOTrainer
from training_config import TRAINING_CONFIG
from ..utils.exceptions import DataPreparationError, TrainingError

def create_data_yaml(dataset_path, num_classes, class_names):
    """創建 YOLO 訓練所需的 data.yaml 文件"""
    yaml_content = f"""
train: {dataset_path}/train/images
val: {dataset_path}/val/images
nc: {num_classes}
names: {class_names}
    """
    
    yaml_path = Path(dataset_path) / 'data.yaml'
    yaml_path.write_text(yaml_content)
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description='YOLO 模型訓練腳本')
    parser.add_argument('--dataset', type=str, required=True, help='數據集路徑')
    parser.add_argument('--classes', type=int, required=True, help='類別數量')
    parser.add_argument('--names', type=str, required=True, help='類別名稱，用逗號分隔')
    args = parser.parse_args()
    
    try:
        # 準備數據集
        print("準備數據集...")
        data_prep = DataPreparation(args.dataset)
        data_prep.setup_directory_structure()
        data_prep.split_dataset(TRAINING_CONFIG['train_ratio'])
        data_prep.verify_dataset()
        
        # 創建 data.yaml
        class_names = [name.strip() for name in args.names.split(',')]
        yaml_path = create_data_yaml(args.dataset, args.classes, class_names)
        
        # 訓練模型
        print("\n開始訓練模型...")
        trainer = YOLOTrainer()
        trainer.setup_model()
        results = trainer.train(str(yaml_path))
        
        # 驗證模型
        print("\n驗證模型性能...")
        val_results = trainer.validate(str(yaml_path))
        
        print("\n訓練完成！")
        print(f"模型保存在: {Path(args.dataset)/'runs'/'train'/'weights'/'best.pt'}")
        
    except (DataPreparationError, TrainingError) as e:
        print(f"錯誤: {str(e)}")
        return 1
        
if __name__ == "__main__":
    main()