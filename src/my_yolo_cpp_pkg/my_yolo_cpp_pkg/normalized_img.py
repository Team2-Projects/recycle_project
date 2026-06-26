import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge

class DistributionNormalizer(Node):
    def __init__(self):
        super().__init__('distribution_normalizer')
        # 토픽 이름이 /image_raw/compressed 인지 확인해주세요 (질문에서는 image_rqw로 적어주심)
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        self.bridge = CvBridge()
        
        # 목표 분포 (학습 데이터의 평균, 표준편차)
        self.target_mean = 130.0
        self.target_std = 70.0

    def listener_callback(self, msg):
        # 1. 압축 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None: return

        # 2. HSV 변환 (uint8 상태에서 변환하는 것이 안전합니다)
        height, width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 3. V 채널(밝기) 정규화
        v_f = v.astype(np.float32)
        curr_mean = np.mean(v_f)
        curr_std = np.std(v_f)
        
        # 목표 분포로 변환
        normalized_v = (v_f - curr_mean) / (curr_std + 1e-5) * self.target_std + self.target_mean
        normalized_v = np.clip(normalized_v, 0, 255).astype(np.uint8)
        
        # 4. 채널 재결합 및 BGR 변환
        hsv_final = cv2.merge([h, s, normalized_v])
        final_img = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
        
        # 5. 결과 출력
        cv2.imshow('Normalized Image', final_img)
        cv2.waitKey(1)
        
        self.get_logger().info(f'Size: {width}x{height} | 원래 Mean: {curr_mean:.2f}, Std: {curr_std:.2f} -> 변환 후 Mean: {np.mean(normalized_v):.2f}, Std: {np.std(normalized_v):.2f}')
def main(args=None):
    rclpy.init(args=args)
    node = DistributionNormalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()