import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge() # ★ bridge 초기화도 잊지 마세요 ★
        self.declare_parameter('conf', 0.5)
        # 모델 경로를 확인하세요
        self.model = YOLO('/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/yolo_8n_trained_1_openvino_model')
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)

        # self.subscription = self.create_subscription(
        #     Image, '/image_raw', self.listener_callback, 10)

  

    def listener_callback(self, msg):
        conf_threshold = self.get_parameter('conf').get_parameter_value().double_value
        #transformed by 8byte int 
        np_arr = np.frombuffer(msg.data, np.uint8)
        #reconstruction img

        # 1. CvBridge를 사용하여 ROS 이미지를 OpenCV(BGR)로 변환
        # frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=1.5, beta=30)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # print(f"이미지 형태 (Shape): {frame_bgr.shape}")
        # # 추론
        results = self.model.predict(source=frame_rgb, imgsz=640, conf=conf_threshold, verbose=False)
        
        # # # 시각화
        res_plotted_rgb = results[0].plot()
        res_plotted_bgr = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO Python Node", res_plotted_bgr)
        # cv2.imshow("img", frame)
        cv2.waitKey(1)

# --------------

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()