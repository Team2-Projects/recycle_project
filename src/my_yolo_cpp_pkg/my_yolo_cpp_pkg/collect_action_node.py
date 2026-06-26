import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.action import ActionServer
from my_yolo_interfaces.action import CollectImages
import os
import time
from cv_bridge import CvBridge
import cv2
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        # 1. 카메라 토픽 구독 (이게 있어야 데이터가 들어옵니다!)
        self.subscription = self.create_subscription(
            CompressedImage, 
            '/image_raw/compressed', 
            self.listener_callback, 
            10
        )
        
        # 2. 액션 서버 생성
        self._action_server = ActionServer(
            self, CollectImages, 'collect_images', self.execute_callback)
            
        self.latest_frame = None

    def listener_callback(self, msg):
        # 최신 이미지를 멤버 변수에 계속 갱신
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def execute_callback(self, goal_handle):
        # 1. Goal에서 파라미터 가져오기
        folder_name = goal_handle.request.folder_name
        start_index = goal_handle.request.start_index
        count = goal_handle.request.count
        
        self.get_logger().info(f'저장 시작: {folder_name}, 시작점: {start_index}, 개수: {count}')
        
        saved_count = 0
        current_idx = start_index
        
        # 저장 디렉토리 생성 (없으면 생성)
        save_dir = '/home/hee/yolo_second_dataset'
        os.makedirs(save_dir, exist_ok=True)
        
        while saved_count < count:
            if self.latest_frame is not None:
                # 2. 전달받은 파일이름과 인덱스 조합
                filename = os.path.join(save_dir, f'{folder_name}', f'img_{current_idx}.jpg')
                cv2.imwrite(filename, self.latest_frame)
                
                saved_count += 1
                current_idx += 1
                
                # Feedback 전송
                feedback_msg = CollectImages.Feedback()
                feedback_msg.current_count = saved_count
                goal_handle.publish_feedback(feedback_msg)
                
                time.sleep(0.5)
        
        goal_handle.succeed()
        return CollectImages.Result(success=True)
def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()