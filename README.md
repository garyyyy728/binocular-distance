# Intel RealSense D435 深度測量系統

這個專案實現了使用 Intel RealSense D435 相機的深度測量功能，結合了立體視覺 SGBM（Semi-Global Block Matching）和 YOLO 物件偵測。

## 功能特點

- 使用 SGBM 進行立體視覺深度測量
- 整合 YOLO 物件偵測
- 即時視覺化深度圖和偵測結果
- Intel RealSense D435 相機整合
- 模組化設計，易於擴展
- 支援自定義 YOLO 模型訓練

## 系統需求

- Intel RealSense D435 相機
- Python 3.8+
- 相依套件（見 requirements.txt）

## 安裝步驟

1. 安裝必要套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 連接 Intel RealSense D435 相機

## 使用方法

### 執行主程式
```bash
python src/main.py
```

### 訓練自定義 YOLO 模型

1. 準備數據集：
   - 在 `dataset` 目錄下創建以下結構：
     ```
     dataset/
     ├── images/    # 所有訓練圖片
     └── labels/    # 對應的標籤文件
     ```

2. 執行訓練：
   ```bash
   python src/detection/train.py --dataset ./dataset --classes 3 --names "class1,class2,class3"
   ```
   
   參數說明：
   - `--dataset`: 數據集路徑
   - `--classes`: 類別數量
   - `--names`: 類別名稱（用逗號分隔）

3. 訓練完成後，模型將保存在 `dataset/runs/train/weights/best.pt`

### 使用訓練好的模型

修改 `src/config/detection_config.py` 中的 `model_path` 為訓練好的模型路徑。

## 專案結構

```
src/
├── config/           # 配置文件
├── camera/           # 相機介面
├── depth/            # 深度計算
├── detection/        # 物件偵測
│   ├── data_preparation.py    # 數據準備工具
│   ├── model_trainer.py       # 模型訓練器
│   ├── training_config.py     # 訓練配置
│   └── train.py              # 訓練腳本
├── utils/            # 工具函數
└── main.py          # 主程式入口
```

## 配置說明

- `camera_config.py`: 相機參數設定
- `depth_config.py`: SGBM 深度計算參數
- `detection_config.py`: YOLO 偵測參數
- `training_config.py`: YOLO 訓練參數

## 訓練數據準備

1. 收集目標物體的圖片
2. 使用標註工具（如 labelimg）標註圖片
3. 確保標籤格式符合 YOLO 格式：
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
4. 將圖片和標籤文件分別放入 `dataset/images` 和 `dataset/labels` 目錄

## 錯誤處理

系統包含完整的錯誤處理機制：
- 相機連接錯誤處理
- 深度計算異常處理
- 物件偵測錯誤處理
- 數據準備錯誤處理
- 模型訓練錯誤處理