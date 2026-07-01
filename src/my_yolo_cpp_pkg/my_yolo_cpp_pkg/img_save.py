import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
from pathlib import Path

class Image_SaveNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        # 저장 경로 설정
        self.save_dir = Path("/media/sf_win_folder/background_img/")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # [수정] 기존 파일들을 확인하여 마지막 번호 가져오기
        # "cap_0001.jpg", "cap_0002.jpg" 형태라 가정
        existing_files = list(self.save_dir.glob("background_*.jpg"))
        self.image_id = len(existing_files)  # 이미 저장된 개수부터 시작
        
        self.save_interval = 2.0  # 2초마다 저장
        self.last_saved_time = 0.0
        
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return

        current_time = time.time()
        if current_time - self.last_saved_time > self.save_interval:
            # [수정] 번호 증가 및 파일명 생성 (4자리 숫자 형식: 0001, 0002...)
            self.image_id += 1
            file_name = f"cap_{self.image_id:06d}.jpg"
            save_path = self.save_dir / file_name
            
            cv2.imwrite(str(save_path), frame)
            self.get_logger().info(f"이미지 저장됨: {save_path}")
            
            self.last_saved_time = current_time

        cv2.imshow("ROS2 Image View", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Image_SaveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()