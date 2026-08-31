import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
import cv2
import numpy as np
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
import time
import openvino as ov  # OpenVINO 라이브러리 임포트

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)

# 변환된 OpenVINO 모델 xml 파일 경로
model_path = '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/0829classify_model_openvino/classify_model.xml'
object_id = {
    'can': 0,
    'paper': 1,
    'plastic': 2,
    'trash': 3,
    'glass_bottle': 4,
    'person': 5
}

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        self.frame_count = 0
        self.is_tracking = False 
        self.declare_parameter('conf_threshold', 0.25)

        self.model = YOLO(
    '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/final_openvino_model',
    task='segment'
)
        
        # [수정] ONNX Runtime 대신 OpenVINO Core로 분류 모델 로드
        self.ov_core = ov.Core()
        self.classify_model_ov = self.ov_core.read_model(model_path)
        # 하드웨어 가속 적용 (CPU 또는 AUTO)
        self.compiled_classify_model = self.ov_core.compile_model(self.classify_model_ov, 'CPU')
        
        # 입력 및 출력 키(Key) 가져오기
        self.input_key = self.compiled_classify_model.input(0)
        self.output_key = self.compiled_classify_model.output(0)
        
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(DetectedObject, '/classified_detected_object_info', 10)
        
        self.srv = self.create_service(SetTracking, 'set_tracking_mode', self.srv_callback)

        
        self.target_idx = 0
        self.pred_class = 0

        # 객체 연속 감지 확인용
        self.prev_cls_id = -1
        self.prev_center = None
        self.same_object_count = 0

        # 동일 객체 판단 기준
        self.required_frames = 3
        # cur_obj - prev_obj < 50.0pixel: same things
        self.center_distance_threshold = 50.0
        # 웹으로 이미지 전달
        self.last_image_publish_time = 0.0

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_pub = self.create_publisher(
            CompressedImage,
            '/yolo/image/compressed',
            image_qos
        )

        self.pr

    def is_same_object(self, cls_id, coord):
        """
        이전 프레임 객체와 현재 객체가 같은 객체인지 판단

        coord = [x_center, y_center, width, height]
        """

        current_center = np.array([
            coord[0],
            coord[1]
        ])

        # 이전 객체가 없는 경우
        if self.prev_center is None:
            self.prev_cls_id = cls_id
            self.prev_center = current_center
            self.same_object_count = 1

            return False

        # 중심 좌표 거리 계산
        distance = np.linalg.norm(
            current_center - self.prev_center
        )

        # 클래스가 같고 중심 좌표가 가까우면
        if (
            cls_id == self.prev_cls_id
            and distance < self.center_distance_threshold
        ):
            self.same_object_count += 1

        else:
            # 다른 객체라고 판단 → 다시 1부터 시작
            self.same_object_count = 1

        # 현재 객체 정보 저장
        self.prev_cls_id = cls_id
        self.prev_center = current_center

        # 3프레임 이상 연속 감지되었는지
        return self.same_object_count >= self.required_frames

    def publish_image(self, frame):
        now = time.time()

        if now - self.last_image_publish_time < 0.2:
            return

        self.last_image_publish_time = now

        success, encoded = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not success:
            return

        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        self.image_pub.publish(msg)

    def srv_callback(self, request, response):
        self.is_tracking = request.enable
        self.get_logger().info(f"🚀 추적 모드 변경: {self.is_tracking}")
        response.success = True
        return response

    def get_closest_to_center(self, boxes):
        centers_x = boxes.xywh[:, 0].tolist()
        distances = [abs(x - 400) for x in centers_x]
        return distances.index(min(distances))

    def listener_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 1 != 0:
            return

        conf_val = self.get_parameter('conf_threshold').get_parameter_value().double_value
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # start_time = time.time()
        msg_data = DetectedObject()

        # # 1. 전처리 (기존과 동일하게 텐서 형태 맞추기)
        # frame_classify = frame.transpose(1, 0, 2)
        # frame_classify = np.expand_dims(frame_classify, axis=0).astype(np.float32)
        
        # 2. [수정] OpenVINO 모델 추론 실행
        result = self.compiled_classify_model({self.input_key: frame.reshape(1,480,640,3)})[self.output_key]

        self.pred_class = np.argmax(result[0])

        if self.pred_class == 0:

            msg_data.id = -1
            msg_data.confidence = 0.0
            msg_data.coord = [0.0, 0.0, 0.0, 0.0]
            msg_data.max_y_up = 0.0

            self.prev_cls_id = -1
            self.prev_center = None
            self.same_object_count = 0
            
        if self.pred_class != 0:
            results = self.model.predict(source=frame, imgsz=640, conf=conf_val, verbose=False)
            res = results[0]
            # self.get_logger().info('box_number = {}'.format(len(res.boxes)))
            confidences = res.boxes.conf.tolist()
            if len(res.boxes) > 0:
                if self.is_tracking:
                    self.target_idx = self.get_closest_to_center(res.boxes)
                else:
                    self.target_idx = confidences.index(max(confidences))
                    
                best_cls_id = int(res.boxes.cls[self.target_idx].item())
                best_name = res.names[best_cls_id]
                best_coord = res.boxes.xywh[self.target_idx].tolist()
                    
                if best_name in object_id:

                    current_cls_id = object_id[best_name]

                    # 현재 객체가 이전 프레임 객체와 같은지 확인
                    is_stable = self.is_same_object(
                        current_cls_id,
                        best_coord
                    )

                    if is_stable:
                        # 3프레임 이상 같은 객체 → 정상 발행
                        msg_data.id = current_cls_id
                        msg_data.confidence = confidences[self.target_idx]
                        msg_data.coord = [float(x) for x in best_coord]
                        msg_data.max_y_up = best_coord[1] - 0.5 * best_coord[3]

                    else:
                        # 아직 3프레임 연속 조건 미충족
                        msg_data.id = -1
                        msg_data.confidence = 0.0
                        msg_data.coord = [0.0, 0.0, 0.0, 0.0]
                        msg_data.max_y_up = 0.0
                else:
                    msg_data.id = -1
                    msg_data.confidence = 0.0
                    msg_data.coord = [0.0, 0.0, 0.0, 0.0]
                    msg_data.max_y_up = 0.0

                    # 연속 객체 정보 초기화
                    self.prev_cls_id = -1
                    self.prev_center = None
                    self.same_object_count = 0
           

        self.publisher_.publish(msg_data)
        self.publish_image(frame)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()