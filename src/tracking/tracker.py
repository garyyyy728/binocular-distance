"""Object tracking implementation"""
import numpy as np
from collections import OrderedDict
import cv2

class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # 儲存追蹤的物體 {ID: centroid}
        self.disappeared = OrderedDict()  # 記錄物體消失的幀數
        self.object_paths = OrderedDict()  # 記錄物體的運動軌跡
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        """註冊新物體"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_paths[self.next_object_id] = [centroid]
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """取消註冊消失的物體"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_paths[object_id]
        
    def update(self, rects):
        """
        更新追蹤狀態
        
        Args:
            rects: 檢測到的物體邊界框列表 [(x1, y1, x2, y2), ...]
        """
        if len(rects) == 0:
            # 所有物體都消失了
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        # 計算當前幀中物體的質心
        centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids[i] = (cx, cy)
            
        if len(self.objects) == 0:
            # 如果沒有追蹤的物體，註冊所有檢測到的物體
            for i in range(len(centroids)):
                self.register(centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # 計算所有物體之間的距離
            distances = np.zeros((len(object_centroids), len(centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(centroids)):
                    distances[i, j] = np.linalg.norm(object_centroids[i] - centroids[j])
                    
            # 找到最佳匹配
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                self.object_paths[object_id].append(centroids[col])
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(centroids))) - used_cols
            
            # 處理消失的物體
            if len(object_centroids) >= len(centroids):
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # 註冊新物體
                for col in unused_cols:
                    self.register(centroids[col])
                    
        return self.objects
        
    def draw_tracks(self, frame):
        """繪製追蹤軌跡"""
        for object_id, path in self.object_paths.items():
            if len(path) > 1:
                # 繪製軌跡
                path_array = np.array(path, dtype=np.int32)
                cv2.polylines(frame, [path_array], False, (0, 255, 0), 2)
                
                # 在當前位置顯示ID
                if object_id in self.objects:
                    current_pos = self.objects[object_id]
                    cv2.putText(frame, f"ID {object_id}",
                              (current_pos[0] - 10, current_pos[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame