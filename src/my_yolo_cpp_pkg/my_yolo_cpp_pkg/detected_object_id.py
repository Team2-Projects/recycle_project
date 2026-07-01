import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
# from std_msgs.msg import Int32  # 객체 ID를 보내기 위한 메시지 타입
from ultralytics import YOLO
import cv2
import numpy as np
from my_yolo_msgs.msg import DetectedObject

object_id = {'can': 0, 'paper': 1, 'plastic': 2}

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.declare_parameter('conf_threshold', 0.5)
        self.model = YOLO('/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/yolo_8n.onnx')
        
        # 1. 구독자 설정
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        
        # 2. 퍼블리셔 설정 (/detected_object_id 토픽으로 ID 발행)
        self.publisher_ = self.create_publisher(DetectedObject, '/detected_object_info', 10)

    def listener_callback(self, msg):
            conf_val = self.get_parameter('conf_threshold').get_parameter_value().double_value
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            results = self.model.predict(source=frame, imgsz=640, conf=conf_val, verbose=False)
            res = results[0]

            msg_data = DetectedObject()

            if len(res.boxes) > 0:
                confidences = res.boxes.conf.tolist()
                max_conf_idx = confidences.index(max(confidences))
                
                best_cls_id = int(res.boxes.cls[max_conf_idx].item())
                best_name = res.names[best_cls_id]
                best_coord = res.boxes.xywh[max_conf_idx].tolist()
                
                # [수정] 감지된 이름이 딕셔너리에 있을 때만 값을 할당하고, 없으면 -1 처리
                if best_name in object_id:
                    msg_data.id = object_id[best_name]
                    msg_data.coord = [float(x) for x in best_coord] # 타입 명시적 변환
                else:
                    msg_data.id = -1
                    msg_data.coord = [0.0, 0.0, 0.0, 0.0]
            else:
                msg_data.id = -1
                msg_data.coord = [0.0, 0.0, 0.0, 0.0]

            self.publisher_.publish(msg_data)
def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()