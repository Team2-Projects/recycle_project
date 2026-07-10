import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
# from std_msgs.msg import Int32  # 객체 ID를 보내기 위한 메시지 타입
from ultralytics import YOLO
import cv2
import numpy as np
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking

object_id = {'can': 0, 'paper': 1, 'plastic': 2}

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        self.frame_count = 0

        self.is_tracking = False # 추적 모드 플래그
        self.declare_parameter('conf_threshold', 0.4)
        self.model = YOLO('/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/transfer_v3_openvino_model')
        # 모델 경로를 확인하세요
        # 1. 패키지의 share 경로를 자동으로 찾음
        # package_share_directory = get_package_share_directory('my_yolo_cpp_pkg')

        # # 2. 모델 경로를 조합
        # model_path = os.path.join(package_share_directory, 'models', 'transfer_v2_openvino_model')

        # # 3. 모델 로드
        # self.model = YOLO(model_path)

        # 1. 구독자 및 퍼블리셔
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(DetectedObject, '/detected_object_info', 10)
        
        # 2. 서비스 서버 (RecycleTrackingNode와 통신)
        self.srv = self.create_service(SetTracking, 'set_tracking_mode', self.srv_callback)

    def srv_callback(self, request, response):
        self.is_tracking = request.enable
        self.get_logger().info(f"🚀 추적 모드 변경: {self.is_tracking}")
        response.success = True
        return response

    def get_closest_to_center(self, boxes):
        """화면 중앙(320)과 가장 가까운 물체의 인덱스를 반환"""
        centers_x = boxes.xywh[:, 0].tolist()
        # 중앙(320)과의 절대 거리 계산
        distances = [abs(x - 320) for x in centers_x]
        return distances.index(min(distances))

    def listener_callback(self, msg):
        self.frame_count += 1

        if self.frame_count % 1 != 0:
            return

        
        conf_val = self.get_parameter('conf_threshold').get_parameter_value().double_value
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
        results = self.model.predict(source=frame, imgsz=640, conf=conf_val, verbose=False)
        res = results[0]
        msg_data = DetectedObject()

        if len(res.boxes) > 0:
            # [수정] 모드에 따라 타겟 선정 방식 변경
            if self.is_tracking:
                target_idx = self.get_closest_to_center(res.boxes)
            else:
                confidences = res.boxes.conf.tolist()
                target_idx = confidences.index(max(confidences))
                
            best_cls_id = int(res.boxes.cls[target_idx].item())
            best_name = res.names[best_cls_id]
            best_coord = res.boxes.xywh[target_idx].tolist()
                
            if best_name in object_id:
                msg_data.id = object_id[best_name]
                msg_data.coord = [float(x) for x in best_coord]
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