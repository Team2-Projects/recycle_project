import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
import openvino as ov

clf_idx = {
    'can': 0,
    'paper': 1,
    'plastic': 2,
    'trash': 3,
    'person': 4
}

# OpenVINO 분류 모델 경로
model_path = (
    '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/'
    '0907_classify_model_openvino/classify_model.xml'
)

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.frame_count = 0
        self.bridge = CvBridge() # ★ bridge 초기화도 잊지 마세요 ★
        self.declare_parameter('conf', 0.25)
        # 모델 경로를 확인하세요
        # 1. 패키지의 share 경로를 자동으로 찾음
        # package_share_directory = get_package_share_directory('my_yolo_cpp_pkg')

        # # 2. 모델 경로를 조합
        # model_path = os.path.join(package_share_directory, 'models', 'transfer_v2_openvino_model')

        # # 3. 모델 로드
        # self.model = YOLO(model_path)
        self.model = YOLO(
    '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/0907_yolo_openvino_model',
    task='segment'
)
        
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)

        # self.subscription = self.create_subscription(
        #     Image, '/image_raw', self.listener_callback, 10)

                # OpenVINO 분류 모델 로드
        self.ov_core = ov.Core()

        self.classify_model_ov = self.ov_core.read_model(
            model_path
        )

        self.compiled_classify_model = (
            self.ov_core.compile_model(
                self.classify_model_ov,
                'CPU'
            )
        )

        # OpenVINO 입출력 키
        self.input_key = self.compiled_classify_model.input(0)
        self.output_key = self.compiled_classify_model.output(0)


    def listener_callback(self, msg):

        conf_threshold = self.get_parameter('conf').get_parameter_value().double_value
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


        classify_result = self.compiled_classify_model(
            {
                self.input_key:
                frame.reshape(1, 480, 640, 3)
            }
        )[self.output_key]

        self.pred_class = np.argmax(
            classify_result [0]
        )

        
        if self.pred_class == 0:
            best_name = None
            best_idx = None
            coord = None

            cv2.imshow("YOLO Python Node", frame)
            cv2.waitKey(10)

        elif self.pred_class == 1:
            results = self.model.predict(source=frame, imgsz=640, conf=conf_threshold, verbose=False)
            res = results[0]
            if len(res.boxes) > 0:
                  # 2. 모든 객체의 신뢰도(conf)를 가져와서 가장 높은 인덱스를 찾음
                  # res.boxes.conf는 텐서 형태이므로 리스트로 변환 후 사용
                confidences = res.boxes.conf.tolist()
                max_conf_idx = confidences.index(max(confidences))

                  # 3. 가장 높은 신뢰도를 가진 객체의 클래스 ID 추출
                best_cls_id = int(res.boxes.cls[max_conf_idx].item())
                best_name = res.names[best_cls_id]
                best_idx = clf_idx[best_name]
                coord = res.boxes.xywh[max_conf_idx].tolist()

                x,y,w,h = coord
              # print(x,y,w,h)
                pt1_x = int(x - (w/2))
                pt1_y = int(y - (h/2))
                pt2_x = int(x + (w/2))
                pt2_y = int(y + (h/2))

                org_x = pt1_x - 5
                org_y = pt1_y - 5
                  
                cv2.rectangle(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 40, 200), 3)
                cv2.putText(frame, best_name, (org_x, org_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, thickness = 3, color = (255, 0, 0))
                cv2.imshow("YOLO Python Node", frame)
                # cv2.imshow("img", frame)
                cv2.waitKey(10)



            else:
                best_name = None
                best_idx = None
                coord = None

                cv2.imshow("YOLO Python Node", frame)
                cv2.waitKey(10)

              # res_plotted_rgb = res.plot()
              # res_plotted_bgr = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)
              # cv2_imshow(res_plotted_bgr)

              # cv2_imshow(frame_bgr)
            
# --------------

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()