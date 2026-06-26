import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge

class BrightnessAnalyzer(Node):
    def __init__(self):
        super().__init__('brightness_analyzer')
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.brightness_values = []
        self.target_count = 100

    def listener_callback(self, msg):
        # 1. 압축 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None: return

        # 2. HSV 변환 및 V 채널(밝기) 추출
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # 3. 평균 밝기 계산 후 저장
        avg_v = np.mean(v_channel)
        self.brightness_values.append(avg_v)
        
        # 4. 100개 도달 시 통계 출력
        if len(self.brightness_values) >= self.target_count:
            mean_val = np.mean(self.brightness_values)
            std_val = np.std(self.brightness_values)
            
            self.get_logger().info(f'--- 통계 결과 (최근 {self.target_count} 프레임) ---')
            self.get_logger().info(f'평균 밝기: {mean_val:.2f}, 표준편차: {std_val:.2f}')
            
            # 리스트 초기화 (또는 슬라이딩 윈도우 방식으로 최신 100개 유지)
            self.brightness_values = [] 

def main(args=None):
    rclpy.init(args=args)
    analyzer = BrightnessAnalyzer()
    rclpy.spin(analyzer)
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()