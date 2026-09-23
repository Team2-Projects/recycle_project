from pathlib import Path

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from ultralytics import YOLO

import cv2
import numpy as np
import time
import math
import openvino as ov

from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from std_srvs.srv import SetBool

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy
)
from PIL import Image

# 실행 위치와 사용자 이름에 의존하지 않도록 ROS 패키지의 모델 폴더를 사용한다.
model_dir = Path(get_package_share_directory('my_yolo_cpp_pkg')) / 'models'
model_path = str(model_dir / '0917_yolo_based(A)_best_openvino' / 'model.xml')


# 객체 ID
object_id = {
    'can': 0,
    'paper': 1,
    'plastic': 2,
    'trash': 3,
    'person': 4
}
object_name = {value: key for key, value in object_id.items()}


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        self.frame_count = 0
        self.is_tracking = False
        self.inference_enabled = True
        self._pause_deadline = None
        self._image_generation = 0
        self.inference_pause_timeout_sec = float(self.declare_parameter(
            'inference_pause_timeout_sec', 5.0).value)
        self.paused_image_hz = float(self.declare_parameter('paused_image_hz', 5.0).value)
        for name in ('inference_pause_timeout_sec', 'paused_image_hz'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name}: 0보다 큰 유한한 값이 필요합니다.')

        # YOLO confidence threshold
        self.declare_parameter('conf_threshold', 0.50)

        # YOLO 모델 로드
        self.model = YOLO(
            str(model_dir / 'yolo_0921_best_openvino_model'),
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
        self.input_layer = self.compiled_classify_model.input(0)
        self.output_layer = self.compiled_classify_model.output(0)

        # 이미지 구독
        self.subscription = self._subscribe_images()

        # 채택 종류는 navigation에서만 결정하며, 화면 비교에만 사용한다.
        self.selected_class_id = -1
        self.selected_class_subscription = self.create_subscription(
            Int32, '/selected_recycle_class', self.selected_class_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # 객체 정보 발행
        self.publisher_ = self.create_publisher(
            DetectedObject,
            '/classified_detected_object_info',
            10
        )
        # 연속 검출 필터 전의 관측: 접근 평균과 수거함 재분류에 클래스 변경도 전달한다.
        self.class_observation_pub = self.create_publisher(
            Detection2DArray, '/yolo/class_observation', 1)

        # 추적 모드 서비스
        self.srv = self.create_service(
            SetTracking,
            'set_tracking_mode',
            self.srv_callback
        )
        self.inference_srv = self.create_service(
            SetBool, 'set_inference_enabled', self.inference_callback)
        self.inference_watchdog = self.create_timer(0.5, self._check_pause_timeout)

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
        self.required_frames = 1

        # 최초 객체 중심으로부터 허용 거리
        self.center_distance_threshold = 50.0

        # 웹과 로컬 화면이 공유할 영상의 발행 시간
        self.last_image_publish_time = None

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
        self.last_detection_log_time = 0.0

    def selected_class_callback(self, msg):
        """새 검출로 채택 값을 덮어쓰지 않고 navigation의 선택·해제만 반영한다."""
        self.selected_class_id = msg.data if msg.data in object_name else -1

    def _subscribe_images(self):
        """최신 영상만 처리하고 모드 전환 전에 대기하던 콜백은 폐기한다."""
        generation = self._image_generation

        def callback(msg):
            if generation == self._image_generation:
                self.listener_callback(msg)

        return self.create_subscription(
            CompressedImage, '/image_raw/compressed', callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

    def _set_inference_enabled(self, enabled):
        if self.inference_enabled == enabled:
            return
        self.inference_enabled = enabled
        self.reset_tracking()
        self.last_image_publish_time = None
        # 전환 이전의 검출과 카메라 수신 큐를 다음 탐색에 재사용하지 않는다.
        self._image_generation += 1
        self.destroy_subscription(self.subscription)
        self.subscription = self._subscribe_images()
        empty = DetectedObject()
        empty.id = -1
        self.publisher_.publish(empty)
        self.get_logger().info(f'분류·YOLO 추론 {"ON" if enabled else "OFF"}')

    def inference_callback(self, request, response):
        """OFF는 주기적으로 갱신해야 유지되며 활성 추적보다 우선하지 않는다."""
        if not request.data and self.is_tracking:
            response.success = False
            response.message = '추적 중에는 추론을 중지할 수 없습니다.'
            return response
        self._pause_deadline = (
            None if request.data else time.monotonic() + self.inference_pause_timeout_sec)
        self._set_inference_enabled(request.data)
        response.success = True
        response.message = 'enabled' if request.data else 'paused'
        return response

    def _check_pause_timeout(self):
        """네비게이션 종료·연결 단절 뒤 영구 중지 상태가 남지 않게 한다."""
        if self._pause_deadline is not None and time.monotonic() >= self._pause_deadline:
            self._pause_deadline = None
            self.get_logger().warning('추론 중지 갱신이 끊겨 분류·YOLO 추론을 재개합니다.')
            self._set_inference_enabled(True)

    def publish_paused_image(self, msg):
        """모델과 전처리는 건너뛰고 낮은 빈도로 PAUSED 카메라 화면만 보낸다."""
        now = time.monotonic()
        if (self.last_image_publish_time is not None
                and now - self.last_image_publish_time < 1.0 / self.paused_image_hz):
            return
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return
            self.publish_image(frame, msg.header, None, None, False)
        except cv2.error as exc:
            self.get_logger().warning(f'중지 중 영상 생성 실패: {exc}')

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
    # 추론 결과를 웹과 로컬 화면에 함께 발행
    # =====================================

    def publish_image(self, frame, header, result, selected_idx, accepted):
        """제어에 사용한 프레임과 결과를 그려 최대 5Hz로 발행한다."""
        now = time.monotonic()
        if (self.last_image_publish_time is not None
                and now - self.last_image_publish_time < 0.2):
            return
        self.last_image_publish_time = now

        # 표시하지 않을 프레임은 그리기와 JPEG 압축도 생략한다.
        try:
            self._draw_detections(frame, result, selected_idx, accepted)
            success, encoded = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                return
            msg = CompressedImage()
            msg.header = header
            msg.format = 'jpeg'
            msg.data = encoded.tobytes()
            self.image_pub.publish(msg)
        except cv2.error as exc:
            # 화면 처리 오류가 다음 프레임의 제어용 검출까지 중단하지 않게 한다.
            self.get_logger().warning(f'검출 영상 생성 실패: {exc}')

    @staticmethod
    def _draw_text(frame, text, origin, scale, thickness=2):
        """배경을 가리지 않고 노란 글자와 얇은 검정 외곽선으로 대비를 확보한다."""
        for color, stroke in (((0, 0, 0), thickness + 2), ((0, 255, 255), thickness)):
            cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, stroke, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_box_label(frame, text, origin, thickness):
        """글자 크기는 유지하고 외곽선까지 영상 안에 들어오도록 배치한다."""
        scale = 0.5
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness + 2)
        height, width = frame.shape[:2]
        x = max(3, min(origin[0], width - text_width - 4))
        y = max(text_height + 3, min(origin[1], height - baseline - 4))
        YoloNode._draw_text(frame, text, (x, y), scale, thickness)

    def _draw_status(self, frame, status, detected_name=None, confidence=None):
        """박스 위치와 관계없이 화면 상단에 현재 검출과 채택 종류를 나란히 표시."""
        selected_name = object_name.get(self.selected_class_id)
        detected = '-' if detected_name is None else f'{detected_name.upper()} {confidence:.2f}'
        selected = '-' if selected_name is None else selected_name.upper()
        mismatch = (detected_name is not None and selected_name is not None
                    and detected_name != selected_name)
        self._draw_text(frame, status + (' | CLASS DIFF' if mismatch else ''), (10, 25), 0.6)
        self._draw_text(frame, f'DETECTED: {detected} | SELECTED: {selected}', (10, 49), 0.6)

    def _draw_detections(self, frame, result, selected_idx, accepted):
        """같은 추론 결과에서 선택된 대상의 박스와 라벨만 하나씩 표시한다."""
        if not self.inference_enabled:
            self._draw_status(frame, 'PAUSED | Inference off')
            return
        detected_name, confidence = None, None
        if result is not None and selected_idx is not None and len(result.boxes) > 0:
            detected_name = result.names[int(result.boxes.cls[selected_idx].item())]
            confidence = float(result.boxes.conf[selected_idx].item())
        mode = 'TRACKING' if self.is_tracking else 'SEARCH'
        if result is None:
            status = 'Background'
        elif len(result.boxes) == 0:
            status = 'No detection'
        elif accepted:
            status = 'Valid detection'
        elif detected_name in object_id:
            status = f'Confirming {self.same_object_count}/{self.required_frames}'
        else:
            status = 'Unsupported class'

        if result is None:
            self._draw_status(frame, f'{mode} | {status}')
            return

        if detected_name is not None:
            height, width = frame.shape[:2]
            thickness = 2
            if accepted:
                color, prefix = (0, 255, 0), 'TARGET '
            elif detected_name in object_id:
                color, prefix = (0, 255, 255), 'PENDING '
            else:
                color, prefix = (160, 160, 160), 'UNSUPPORTED '
            x, y, w, h = result.boxes.xywh[selected_idx].tolist()
            left = max(0, min(width - 1, int(x - w / 2)))
            top = max(0, min(height - 1, int(y - h / 2)))
            right = max(0, min(width - 1, int(x + w / 2)))
            bottom = max(0, min(height - 1, int(y + h / 2)))
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness,
                          lineType=cv2.LINE_AA)
            self._draw_box_label(frame, f'{prefix}{detected_name} {confidence:.2f}',
                                 (left, max(80, top - 8)), thickness)
        # 상단은 마지막에 그려 박스가 상태 문구를 덮지 않게 한다.
        self._draw_status(frame, f'{mode} | {status}', detected_name, confidence)


    # =====================================
    # 추적 모드 서비스
    # =====================================

    def srv_callback(self, request, response):

        if request.enable:
            self._pause_deadline = None
            self._set_inference_enabled(True)
        # 모드 전환 전의 감지 횟수를 다음 최초 탐색에 재사용하지 않는다.
        if self.is_tracking != request.enable:
            self.reset_tracking()
        self.is_tracking = request.enable

        self.get_logger().info(
            f"추적 모드 변경: {self.is_tracking}"
        )

        response.success = True

        return response


    # =====================================
    # 화면 중앙 x=320에 가장 가까운 객체 선택
    # =====================================

    def get_closest_to_center(self, boxes):

        centers_x = boxes.xywh[:, 0].tolist()

        distances = [
            abs(x - 320)
            for x in centers_x
        ]

        return distances.index(
            min(distances)
        )


    # =====================================
    # 이미지 콜백
    # =====================================

    def listener_callback(self, msg):

        if not self.inference_enabled:
            self.publish_paused_image(msg)
            return

        # Pi 촬영 시각과 분리한 PC 수신 시각. 추론을 시작하기 전에 기록한다.
        received_at = self.get_clock().now().to_msg()

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
        res, selected_idx = None, None

        # =====================================
        # 최초 탐색에만 사전 분류를 적용하고, 추적 중에는 YOLO를 바로 실행한다.
        # =====================================
        if not self.is_tracking:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_classify = Image.fromarray(frame_rgb).resize((640, 480))

            # YOLO 방식으로 위/아래 80 pixel씩 padding
            canvas = Image.new("RGB", (640, 640), (114, 114, 114))
            canvas.paste(frame_classify, (0, 80))
            frame_classify = np.array(canvas, dtype=np.float32)
            frame_classify /= 255.0
            frame_classify = np.transpose(frame_classify, (2, 0, 1))
            frame_classify = np.expand_dims(frame_classify, axis=0)

            result = self.compiled_classify_model([frame_classify])[self.output_layer]
            self.pred_class = np.argmax(result[0])

        # =====================================
        # Background
        # =====================================

        if not self.is_tracking and self.pred_class == 0:

            now = time.time()

            if now - self.last_detection_log_time >= 2.0:

                self.get_logger().info(
                    f'물체를 발견하지 못했습니다. result[0]: {result[0]}'
                )

                self.last_detection_log_time = now

            msg_data.id = -1
            msg_data.confidence = 0.0
            msg_data.coord = [
                0.0,
                0.0,
                0.0,
                0.0
            ]

            msg_data.min_y = 0.0

            # 객체 안정성 정보 초기화
            self.reset_tracking()

        # =====================================
        # 추적 중이거나 사전 분류에서 Object 판정 → YOLO 실행
        # =====================================

        elif self.is_tracking or self.pred_class == 1:
   

            now = time.time()

            if not self.is_tracking and now - self.last_detection_log_time >= 2.0:

                self.get_logger().info(
                    f'물체를 발견하였습니다. result[0]: {result[0]}'
                )

                self.last_detection_log_time = now

            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=conf_val,
                verbose=False
            )

            res = results[0]




            # =================================
            # YOLO 객체 발견
            # =================================

            if len(res.boxes) > 0:

                confidences = res.boxes.conf.tolist()
                coords = res.boxes.xywh.tolist()
                y_list = []

                top_y_list = [
                    coords[i][1] - coords[i][3] / 2
                    for i in range(len(coords))
                ]

                min_y = min(top_y_list)
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

                selected_idx = self.target_idx

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

                    observation = Detection2D()
                    observation.header = msg.header
                    center = observation.bbox.center.position
                    center.x, center.y = map(float, best_coord[:2])
                    observation.bbox.size_x, observation.bbox.size_y = map(float, best_coord[2:])
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = best_name
                    hypothesis.hypothesis.score = float(confidences[self.target_idx])
                    observation.results = [hypothesis]
                    batch = Detection2DArray()
                    batch.header.stamp = received_at
                    batch.header.frame_id = msg.header.frame_id
                    batch.detections = [observation]
                    self.class_observation_pub.publish(batch)

                    # =========================
                    # 최초 탐색만 연속 감지를 확인하고, 추적 중 재검출은 즉시 전달한다.
                    # =========================

                    is_stable = (
                        self.is_tracking or self.is_same_object(
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

                        msg_data.min_y = min_y 

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

                        msg_data.min_y = 0.0

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

                    msg_data.min_y = 0.0

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

                msg_data.min_y = 0.0

                # 객체가 끊겼으므로 초기화
                self.reset_tracking()

        # =====================================
        # 객체 정보 발행
        # =====================================

        self.publisher_.publish(
            msg_data
        )

        # 제어 정보부터 전달하고, 같은 프레임의 결과를 화면에 표시한다.
        self.publish_image(
            frame, msg.header, res, selected_idx, msg_data.id >= 0
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
