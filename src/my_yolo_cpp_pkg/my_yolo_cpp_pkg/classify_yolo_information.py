import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO

import cv2
import numpy as np
import time
import openvino as ov

from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)


# OpenVINO 분류 모델 경로
model_path = (
    '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/'
    '0829classify_model_openvino/classify_model.xml'
)


# 객체 ID
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

        # YOLO confidence threshold
        self.declare_parameter('conf_threshold', 0.50)

        # YOLO 모델 로드
        self.model = YOLO(
            '/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/final_openvino_model',
            task='segment'
        )

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

        # 이미지 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )

        # 객체 정보 발행
        self.publisher_ = self.create_publisher(
            DetectedObject,
            '/classified_detected_object_info',
            10
        )

        # 추적 모드 서비스
        self.srv = self.create_service(
            SetTracking,
            'set_tracking_mode',
            self.srv_callback
        )

        self.target_idx = 0
        self.pred_class = 0

        # ==============================
        # 최초 발견 객체 기준 안정성 확인
        # ==============================

        # 최초 발견 객체의 클래스 ID
        self.first_cls_id = -1

        # 최초 발견 객체의 중심 좌표
        self.first_center = None

        # 동일 객체 연속 감지 횟수
        self.same_object_count = 0

        # 총 x프레임 동안 유지되어야 함
        self.required_frames = 2

        # 최초 객체 중심으로부터 허용 거리
        self.center_distance_threshold = 50.0

        # 웹 이미지 발행 시간
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

    # =====================================
    # 객체 안정성 정보 초기화
    # =====================================

    def reset_tracking(self):

        self.first_cls_id = -1
        self.first_center = None
        self.same_object_count = 0


    # =====================================
    # 최초 발견 객체 기준 동일 객체 확인
    # =====================================

    def is_same_object(self, cls_id, coord):

        """
        최초 발견 객체를 기준으로 판단

        조건:
        1. 클래스가 동일해야 함
        2. 최초 객체 중심으로부터
           center_distance_threshold 이내여야 함
        3. 위 조건을 required_frames 만큼
           연속으로 만족해야 함
        """

        current_center = np.array([
            coord[0],
            coord[1]
        ])

        # ---------------------------------
        # 첫 번째 객체 발견
        # ---------------------------------

        if self.first_center is None:

            # 최초 객체 정보 저장
            self.first_cls_id = cls_id
            self.first_center = current_center

            # 첫 번째 프레임
            self.same_object_count = 1

            return False

        # ---------------------------------
        # 최초 객체와 현재 객체 거리 계산
        # ---------------------------------

        # distance = np.linalg.norm(
        #     current_center - self.first_center
        # )

        # ---------------------------------
        # 최초 객체 기준 동일성 판단
        # ---------------------------------

        if (
            cls_id == self.first_cls_id
            # and distance < self.center_distance_threshold
        ):

            # 동일 객체로 판단
            self.same_object_count += 1

        else:

            # 다른 객체로 판단
            # 현재 객체를 새로운 최초 객체로 등록
            self.first_cls_id = cls_id
            self.first_center = current_center

            # 다시 첫 번째 프레임부터 시작
            self.same_object_count = 1

        # ---------------------------------
        # x프레임 이상 유지 여부
        # ---------------------------------

        return (
            self.same_object_count >= self.required_frames
        )


    # =====================================
    # 웹용 이미지 발행
    # =====================================

    def publish_image(self, frame):

        now = time.time()

        # 0.2초마다 이미지 발행
        if now - self.last_image_publish_time < 0.2:
            return

        self.last_image_publish_time = now

        success, encoded = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not success:
            return

        msg = CompressedImage()

        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        self.image_pub.publish(msg)


    # =====================================
    # 추적 모드 서비스
    # =====================================

    def srv_callback(self, request, response):

        self.is_tracking = request.enable

        self.get_logger().info(
            f"추적 모드 변경: {self.is_tracking}"
        )

        response.success = True

        return response


    # =====================================
    # 화면 기준 x=400에 가장 가까운 객체 선택
    # =====================================

    def get_closest_to_center(self, boxes):

        centers_x = boxes.xywh[:, 0].tolist()

        distances = [
            abs(x - 400)
            for x in centers_x
        ]

        return distances.index(
            min(distances)
        )


    # =====================================
    # 이미지 콜백
    # =====================================

    def listener_callback(self, msg):

        self.frame_count += 1

        if self.frame_count % 1 != 0:
            return

        # YOLO confidence
        conf_val = (
            self.get_parameter('conf_threshold')
            .get_parameter_value()
            .double_value
        )

        # CompressedImage → OpenCV 이미지
        np_arr = np.frombuffer(
            msg.data,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        # 발행할 메시지
        msg_data = DetectedObject()

        # =====================================
        # 1단계: 분류 모델
        # =====================================

        result = self.compiled_classify_model(
            {
                self.input_key:
                frame.reshape(1, 480, 640, 3)
            }
        )[self.output_key]

        self.pred_class = np.argmax(
            result[0]
        )

        # =====================================
        # Background
        # =====================================

        if self.pred_class == 100:

       
            msg_data.id = 1
            msg_data.confidence = 0.0
            msg_data.coord = [
                0.0,
                0.0,
                0.0,
                0.0
            ]

            msg_data.max_y_up = 0.0

            # 객체 안정성 정보 초기화
            self.reset_tracking()

        # =====================================
        # Object 존재 → YOLO 실행
        # =====================================

        else:

            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=conf_val,
                verbose=False
            )

            res = results[0]

            confidences = res.boxes.conf.tolist()

            # =================================
            # YOLO 객체 발견
            # =================================

            if len(res.boxes) > 0:

                # -----------------------------
                # 추적 모드
                # -----------------------------

                if self.is_tracking:

                    self.target_idx = (
                        self.get_closest_to_center(
                            res.boxes
                        )
                    )

                # -----------------------------
                # 일반 모드
                # -----------------------------

                else:

                    self.target_idx = (
                        confidences.index(
                            max(confidences)
                        )
                    )

                # 선택된 객체 정보
                best_cls_id = int(
                    res.boxes.cls[
                        self.target_idx
                    ].item()
                )

                best_name = res.names[
                    best_cls_id
                ]

                best_coord = (
                    res.boxes.xywh[
                        self.target_idx
                    ].tolist()
                )

                # =============================
                # 등록된 객체인지 확인
                # =============================

                if best_name in object_id:

                    current_cls_id = object_id[
                        best_name
                    ]

                    # =========================
                    # 최초 객체 기준 안정성 확인
                    # =========================

                    is_stable = (
                        self.is_same_object(
                            current_cls_id,
                            best_coord
                        )
                    )

                    # -------------------------
                    # x프레임 조건 만족
                    # -------------------------

                    if is_stable:

                        msg_data.id = current_cls_id

                        msg_data.confidence = (
                            confidences[
                                self.target_idx
                            ]
                        )

                        msg_data.coord = [
                            float(x)
                            for x in best_coord
                        ]

                        msg_data.max_y_up = (
                            best_coord[1]
                            - 0.5 * best_coord[3]
                        )

                    # -------------------------
                    # 아직 x프레임 미만
                    # -------------------------

                    else:

                        msg_data.id = -1
                        msg_data.confidence = 0.0

                        msg_data.coord = [
                            0.0,
                            0.0,
                            0.0,
                            0.0
                        ]

                        msg_data.max_y_up = 0.0

                # =================================
                # object_id에 없는 클래스
                # =================================

                else:

                    msg_data.id = -1
                    msg_data.confidence = 0.0

                    msg_data.coord = [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]

                    msg_data.max_y_up = 0.0

                    self.reset_tracking()

            # =====================================
            # YOLO 객체 없음
            # =====================================

            else:

                msg_data.id = -1
                msg_data.confidence = 0.0

                msg_data.coord = [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]

                msg_data.max_y_up = 0.0

                # 객체가 끊겼으므로 초기화
                self.reset_tracking()

        # =====================================
        # 객체 정보 발행
        # =====================================

        self.publisher_.publish(
            msg_data
        )

        # 웹용 이미지 발행
        self.publish_image(
            frame
        )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = YoloNode()

    rclpy.spin(
        node
    )

    rclpy.shutdown()


if __name__ == '__main__':
    main()