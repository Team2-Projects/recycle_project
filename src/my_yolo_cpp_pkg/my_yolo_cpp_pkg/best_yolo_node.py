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
    '0909yolo_based(A)_best_openvino/model.xml'
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

        conf_threshold = (
            self.get_parameter('conf')
            .get_parameter_value()
            .double_value
        )

        np_arr = np.frombuffer(msg.data, np.uint8)

        # --------------------------------------------------------
        # 원본 frame
        # YOLO용
        # --------------------------------------------------------

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )


        # ========================================================
        # 분류 모델 입력
        # ========================================================

        frame_classify = frame

        # 640 x 640 canvas 생성
        canvas = np.full(
            (640, 640, 3),
            114,
            dtype=np.uint8
        )

        # 위에서부터 80 pixel padding
        canvas[80:560, :, :] = frame_classify

        frame_classify = canvas.astype(
            np.float32
        )

        # 0~1 정규화
        frame_classify /= 255.0

        # HWC -> NCHW
        frame_classify = np.transpose(
            frame_classify,
            (2, 0, 1)
        )

        # Batch dimension
        frame_classify = np.expand_dims(
            frame_classify,
            axis=0
        )

        # ========================================================
        # OpenVINO 분류
        # ========================================================

        classify_result = self.compiled_classify_model(
            {
                self.input_key: frame_classify
            }
        )[self.output_key]

        self.pred_class = np.argmax(
            classify_result[0]
        )


        # ========================================================
        # Background
        # ========================================================

        if self.pred_class == 0:

            best_name = None
            best_idx = None
            coord = None

            cv2.imshow(
                "YOLO Python Node",
                frame
            )
            cv2.waitKey(10)


        # ========================================================
        # Object
        # ========================================================

        elif self.pred_class == 1:

            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=conf_threshold,
                verbose=False
            )

            res = results[0]

            if len(res.boxes) > 0:

                confidences = res.boxes.conf.tolist()

                max_conf_idx = confidences.index(
                    max(confidences)
                )

                best_cls_id = int(
                    res.boxes.cls[max_conf_idx].item()
                )

                best_name = res.names[best_cls_id]

                best_idx = clf_idx[best_name]

                coord = res.boxes.xywh[
                    max_conf_idx
                ].tolist()

                x, y, w, h = coord

                pt1_x = int(x - (w / 2))
                pt1_y = int(y - (h / 2))
                pt2_x = int(x + (w / 2))
                pt2_y = int(y + (h / 2))

                org_x = pt1_x - 5
                org_y = pt1_y - 5

                cv2.rectangle(
                    frame,
                    (pt1_x, pt1_y),
                    (pt2_x, pt2_y),
                    (0, 40, 200),
                    3
                )

                cv2.putText(
                    frame,
                    best_name,
                    (org_x, org_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    3
                )

                cv2.imshow(
                    "YOLO Python Node",
                    frame
                )
                cv2.waitKey(10)

            else:

                best_name = None
                best_idx = None
                coord = None

                cv2.imshow(
                    "YOLO Python Node",
                    frame
                )
                cv2.waitKey(10)
            
# --------------

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()