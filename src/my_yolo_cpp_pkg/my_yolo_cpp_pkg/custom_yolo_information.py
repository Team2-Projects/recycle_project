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

# 변환된 OpenVINO 모델 xml 파일 경로
model_path = '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/classify_model_openvino/classify_model.xml'
obj_detection_model_path = '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/custom_object_detection_model.openvino/model.xml'
object_id = {'can': 0, 'paper': 1, 'plastic': 2}

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        self.frame_count = 0
        self.is_tracking = False 
        self.declare_parameter('conf_threshold', 0.4)
        
        
        
        # [수정] ONNX Runtime 대신 OpenVINO Core로 분류 모델, obj_detection 로드
        self.ov_core = ov.Core()

        self.obj_model = self.ov_core.compile_model(model=obj_detection_model_path, device_name="CPU")
        self.classify_model_ov = self.ov_core.read_model(model_path)
        # 하드웨어 가속 적용 (CPU 또는 AUTO)
        self.compiled_classify_model = self.ov_core.compile_model(self.classify_model_ov, 'CPU')
        
        # 입력 및 출력 키(Key) 가져오기
        self.input_key = self.compiled_classify_model.input(0)
        self.output_key = self.compiled_classify_model.output(0)

        self.input_layer = self.obj_model.input(0)
        self.output_layer = self.obj_model.output(0)
        
        self.processing_times = []
        
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(DetectedObject, '/custom_detected_object_info', 10)
        
        self.srv = self.create_service(SetTracking, 'set_tracking_mode', self.srv_callback)

        self.target_idx = 0;
        self.pred_class = 0;
    def srv_callback(self, request, response):
        self.is_tracking = request.enable
        self.get_logger().info(f"🚀 추적 모드 변경: {self.is_tracking}")
        response.success = True
        return response

    def get_closest_to_center(self, boxes):
        centers_x = boxes.xywh[:, 0].tolist()
        distances = [abs(x - 320) for x in centers_x]
        return distances.index(min(distances))

    def get_max_pixel_index(self, grid_2d):
        """
        2차원 격자에서 가장 높은 픽셀값을 가지는 인덱스(ix, iy)를 반환하는 함수
        """
        # 1. 2차원 배열에서 가장 큰 값의 1차원 평탄화된 인덱스 찾기
        flat_index = np.argmax(grid_2d)

        # 2. 1차원 인덱스를 원래 2차원 형태(shape)의 좌표(y, x)로 변환
        iy, ix = np.unravel_index(flat_index, grid_2d.shape)

        return int(ix), int(iy)

    def listener_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 1 != 0:
            return

        conf_val = self.get_parameter('conf_threshold').get_parameter_value().double_value
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        start_time = time.time()
        msg_data = DetectedObject()

        # 1. 전처리 (기존과 동일하게 텐서 형태 맞추기)
        frame_classify = frame.transpose(1, 0, 2)
        frame_classify = np.expand_dims(frame_classify, axis=0).astype(np.float32)
        
        # 2. [수정] OpenVINO 모델 추론 실행
        result = self.compiled_classify_model({self.input_key: frame_classify})[self.output_key]

        self.pred_class = np.argmax(result[0])

        if self.pred_class == 3:
            
            msg_data.id = -1
            msg_data.confidence = 0.0
            msg_data.coord = [0.0, 0.0, 0.0, 0.0]
            self.publisher_.publish(msg_data)
        if self.pred_class != 3:
            pred = compiled_model([frame.reshape(1,640,480,3)])[output_layer][0]
            pred = pred.transpose(1,0,2)
            ix, iy = get_max_pixel_index(pred[:,:,4])
            # self.get_logger().info('box_number = {}'.format(len(res.boxes)))
            confidence = pred[iy,ix,4]
            if len(confidence) > 0.1:
                
                msg_data.id = np.argmax(pred_data[5:])
                msg_data.confidence = pred_data[4]
                msg_data.coord = [pred_data[0]*640, pred_data[1]*480, pred_data[2]*640, pred_data[3]*480]
            else:
                msg_data.id = -1
                msg_data.confidence = 0.0
                msg_data.coord = [0.0, 0.0, 0.0, 0.0]

        self.publisher_.publish(msg_data)

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.processing_times.append(elapsed_time)

        current_count = len(self.processing_times)

        if current_count % 10 == 0:
            total_elapsed = sum(self.processing_times)
            avg_time = total_elapsed / current_count
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
            self.get_logger().info('pred_class = {}'.format(self.pred_class))
            self.get_logger().info(
                f"📊 [누적 {current_count}개] 총 소요 시간: {total_elapsed:.2f}초 | "
                f"평균 처리 시간: {avg_time * 1000:.2f}ms | 평균 FPS: {fps:.2f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()