"""Main application interface"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from ..tracking.tracker import ObjectTracker
from ..config.model_config import TRACKER_CONFIG
from .model_manager import ModelManager

class AppInterface:
    def __init__(self):
        self.model_manager = ModelManager()
        self.tracker = ObjectTracker(
            max_disappeared=TRACKER_CONFIG['max_disappeared'],
            max_distance=TRACKER_CONFIG['max_distance']
        )
        self.setup_gui()
        
    def setup_gui(self):
        """設置圖形界面"""
        self.root = tk.Tk()
        self.root.title("深度測量與目標追蹤系統")
        
        # 模型選擇區域
        model_frame = ttk.LabelFrame(self.root, text="模型選擇", padding="5")
        model_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(model_frame, text="模型類型:").grid(row=0, column=0, padx=5, pady=5)
        self.model_type_var = tk.StringVar()
        self.model_type_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_type_var,
            values=list(self.model_manager.supported_models.keys())
        )
        self.model_type_combo.grid(row=0, column=1, padx=5, pady=5)
        self.model_type_combo.bind('<<ComboboxSelected>>', self.update_model_list)
        
        ttk.Label(model_frame, text="模型:").grid(row=1, column=0, padx=5, pady=5)
        self.model_name_var = tk.StringVar()
        self.model_name_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_name_var
        )
        self.model_name_combo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(
            model_frame,
            text="載入模型",
            command=self.load_selected_model
        ).grid(row=2, column=0, columnspan=2, pady=10)
        
        # 追蹤控制區域
        tracking_frame = ttk.LabelFrame(self.root, text="追蹤控制", padding="5")
        tracking_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.show_tracks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            tracking_frame,
            text="顯示追蹤軌跡",
            variable=self.show_tracks_var
        ).grid(row=0, column=0, padx=5, pady=5)
        
        self.show_distance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            tracking_frame,
            text="顯示距離資訊",
            variable=self.show_distance_var
        ).grid(row=1, column=0, padx=5, pady=5)
        
        # 狀態欄
        self.status_var = tk.StringVar(value="就緒")
        status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        status_label.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
    def update_model_list(self, event=None):
        """更新模型列表"""
        model_type = self.model_type_var.get()
        if model_type in self.model_manager.supported_models:
            models = self.model_manager.supported_models[model_type]
            self.model_name_combo['values'] = list(models.keys())
            
    def load_selected_model(self):
        """載入選擇的模型"""
        try:
            model_type = self.model_type_var.get()
            model_name = self.model_name_var.get()
            
            if not model_type or not model_name:
                self.status_var.set("錯誤：請選擇模型類型和模型")
                return
                
            self.model_manager.load_model(model_type, model_name)
            self.status_var.set(f"成功載入模型：{model_name}")
            
        except Exception as e:
            self.status_var.set(f"錯誤：{str(e)}")
            
    def calculate_distance(self, depth_map, bbox):
        """計算物體到相機的距離"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = depth_map[y1:y2, x1:x2]
        # 使用中位數來減少噪聲影響
        distance = np.median(roi)
        return distance
        
    def process_frame(self, frame, depth_map, detections):
        """處理每一幀圖像"""
        # 獲取檢測框
        bboxes = []
        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            if conf > TRACKER_CONFIG['min_confidence']:
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                
        # 更新追蹤器
        objects = self.tracker.update(bboxes)
        
        # 繪製追蹤結果
        if self.show_tracks_var.get():
            frame = self.tracker.draw_tracks(frame)
            
        # 顯示距離信息
        if self.show_distance_var.get():
            for object_id, centroid in objects.items():
                # 找到對應的邊界框
                for bbox in bboxes:
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    if abs(cx - centroid[0]) < 10 and abs(cy - centroid[1]) < 10:
                        distance = self.calculate_distance(depth_map, bbox)
                        cv2.putText(
                            frame,
                            f"ID {object_id}: {distance:.2f}m",
                            (centroid[0] - 10, centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        break
                        
        return frame
        
    def run(self):
        """運行主循環"""
        self.root.mainloop()