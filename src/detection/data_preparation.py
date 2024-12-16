"""Data preparation utilities for YOLO training"""
import os
import shutil
from pathlib import Path
import cv2
import random
from ..utils.exceptions import DataPreparationError

class DataPreparation:
    def __init__(self, dataset_path):
        """
        初始化數據準備工具
        
        Args:
            dataset_path (str): 數據集根目錄路徑
        """
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / 'train'
        self.val_path = self.dataset_path / 'val'
        self.images_path = self.dataset_path / 'images'
        self.labels_path = self.dataset_path / 'labels'
        
    def setup_directory_structure(self):
        """建立訓練所需的目錄結構"""
        try:
            # 創建必要的目錄
            for path in [self.train_path, self.val_path,
                        self.train_path / 'images', self.train_path / 'labels',
                        self.val_path / 'images', self.val_path / 'labels']:
                path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DataPreparationError(f"創建目錄結構失敗: {str(e)}")
            
    def split_dataset(self, train_ratio=0.8):
        """
        分割數據集為訓練集和驗證集
        
        Args:
            train_ratio (float): 訓練集比例 (0-1)
        """
        try:
            # 獲取所有圖片文件
            image_files = list(self.images_path.glob('*.jpg')) + \
                         list(self.images_path.glob('*.png'))
            
            # 隨機打亂
            random.shuffle(image_files)
            
            # 計算分割點
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            # 複製文件到對應目錄
            for files, target_dir in [(train_files, self.train_path),
                                    (val_files, self.val_path)]:
                for img_path in files:
                    # 複製圖片
                    shutil.copy2(img_path, target_dir / 'images' / img_path.name)
                    
                    # 複製對應的標籤文件
                    label_path = self.labels_path / f"{img_path.stem}.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, target_dir / 'labels' / f"{img_path.stem}.txt")
                        
        except Exception as e:
            raise DataPreparationError(f"分割數據集失敗: {str(e)}")
            
    def verify_dataset(self):
        """驗證數據集的完整性"""
        try:
            for split in ['train', 'val']:
                split_path = self.dataset_path / split
                images = list((split_path / 'images').glob('*.jpg')) + \
                        list((split_path / 'images').glob('*.png'))
                labels = list((split_path / 'labels').glob('*.txt'))
                
                print(f"{split} 集合統計:")
                print(f"  - 圖片數量: {len(images)}")
                print(f"  - 標籤數量: {len(labels)}")
                
                # 檢查是否每張圖片都有對應的標籤
                missing_labels = []
                for img in images:
                    label_path = split_path / 'labels' / f"{img.stem}.txt"
                    if not label_path.exists():
                        missing_labels.append(img.name)
                
                if missing_labels:
                    print(f"  - 警告: {len(missing_labels)} 張圖片缺少標籤文件")
                    
        except Exception as e:
            raise DataPreparationError(f"驗證數據集失敗: {str(e)}")