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
        self.declare_parameter('save_dir', '/home/hee/yolo_second_dataset')
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
        # 1. 파라미터로부터 저장 경로 가져오기
        save_base_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        
        folder_name = goal_handle.request.folder_name
        start_index = goal_handle.request.start_index
        count = goal_handle.request.count
        
        # 폴더 경로 조합 (base_dir/folder_name)
        full_save_dir = os.path.join(save_base_dir, folder_name)
        os.makedirs(full_save_dir, exist_ok=True)
        
        self.get_logger().info(f'저장 경로: {full_save_dir}')
        
        saved_count = 0
        current_idx = start_index
        
        while saved_count < count:
            if self.latest_frame is not None:
                # 2. 파일 저장
                filename = os.path.join(full_save_dir, f'img_{current_idx}.jpg')
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