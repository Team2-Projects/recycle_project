import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        # 모델 경로를 확인하세요
        self.model = YOLO('/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/best.onnx')
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)

  

    def listener_callback(self, msg):
      
        #transformed by 8byte int 
        np_arr = np.frombuffer(msg.data, np.uint8)
        #reconstruction img
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # # 추론
        results = self.model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        
        # # # 시각화
        res_plotted = results[0].plot()
        cv2.imshow("YOLO Python Node", res_plotted)
        # cv2.imshow("img", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()