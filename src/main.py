"""Main application entry point"""
import cv2
import sys
from camera.realsense_camera import RealSenseCamera
from depth.sgbm_processor import SGBMProcessor
from interface.app_interface import AppInterface
from utils.visualization import create_depth_visualization, show_images
from utils.exceptions import CameraError, DepthProcessingError, DetectionError

def main():
    print("初始化深度測量系統...")
    
    try:
        # 初始化組件
        camera = RealSenseCamera()
        depth_processor = SGBMProcessor()
        app = AppInterface()
        
        print("正在啟動相機...")
        camera.start()
        print("相機啟動成功！")
        print("\n按 'q' 鍵退出程式")
        
        while True:
            try:
                # 獲取相機幀
                left_ir, right_ir, color_image = camera.get_frames()
                
                # 計算深度
                disparity, disparity_normalized = depth_processor.compute_depth(
                    left_ir, right_ir
                )
                
                # 執行目標檢測
                current_model = app.model_manager.get_current_model()
                if current_model:
                    detections = current_model.detect(color_image)
                    
                    # 處理檢測結果
                    processed_frame = app.process_frame(
                        color_image.copy(),
                        disparity,
                        detections
                    )
                    
                    # 創建可視化
                    visualization = create_depth_visualization(
                        disparity_normalized,
                        processed_frame,
                        detections
                    )
                    
                    # 顯示結果
                    show_images(left_ir, right_ir, visualization)
                    
                # 更新GUI
                app.root.update()
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n正在關閉程式...")
                    break
                    
            except (CameraError, DepthProcessingError, DetectionError) as e:
                print(f"錯誤: {str(e)}")
                break
                
    except Exception as e:
        print(f"嚴重錯誤: {str(e)}")
        sys.exit(1)
        
    finally:
        if 'camera' in locals():
            camera.stop()
        cv2.destroyAllWindows()
        print("程式已安全關閉")

if __name__ == "__main__":
    main()