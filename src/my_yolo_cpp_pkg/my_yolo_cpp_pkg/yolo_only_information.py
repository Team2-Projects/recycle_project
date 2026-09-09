"""YOLO-only, latest-frame perception node. No pre-classifier import/load/inference.

Keeps the legacy DetectedObject + SetTracking interfaces. Do NOT run alongside
classified_object_info_node: both would publish/control the same interfaces.
"""
from pathlib import Path
from threading import Lock
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from ultralytics import YOLO

from my_yolo_cpp_pkg.vision_selection import (
    CLASS_IDS, Candidate, StableSelector, build_model_to_project_class_map,
)


def _model_search_roots():
    """Search roots for a relative model_path, independent of the build flag.

    1. Installed package share: models/ is installed by setup.py, so a plain
       `colcon build` works. With --symlink-install this points at the source.
    2. Source package directory: covers a --symlink-install layout where the
       Python module itself is symlinked back into src/.

    The workspace's absolute location never matters; only the repo-internal
    layout does, so every teammate uses the same YAML.
    """
    roots = []
    try:
        from ament_index_python.packages import get_package_share_directory
        roots.append(Path(get_package_share_directory('my_yolo_cpp_pkg')))
    except Exception:
        pass  # Not sourced yet, or running the module directly.
    roots.append(Path(__file__).resolve().parents[1])
    return roots


def resolve_model_path(value: str) -> Path:
    """Resolve model_path without absolute paths and without a required flag.

    An absolute path is still accepted verbatim. A relative path is resolved
    against the package, never against the shell's current directory, so the
    node cannot silently pick up another project's model.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError('model_path must be a nonempty path')
    requested = Path(value).expanduser()
    if requested.is_absolute():
        candidates = [requested]
    else:
        candidates = [root / requested for root in _model_search_roots()]
    checked = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        checked.append(str(resolved))
    raise FileNotFoundError(
        'YOLO model not found. Checked: ' + ', '.join(checked) + '. '
        'Keep the export under my_yolo_cpp_pkg/models/ (it is installed by '
        'setup.py), rebuild the package, and re-source install/setup.bash. '
        'An absolute model_path also works. No model is downloaded automatically.'
    )


class YoloOnlyNode(Node):
    def __init__(self):
        super().__init__('yolo_only_node')
        defaults = {
            'model_path': 'models/0907_yolo_openvino_model',
            'model_task': 'segment', 'image_topic': '/image_raw/compressed',
            'conf_threshold': 0.50, 'imgsz': 640, 'infer_fps': 5.0,
            'expected_width': 640, 'expected_height': 480,
            'align_reference_x': 350.0, 'approach_stop_lower_y': 430.0,
            'required_frames': 2, 'allowed_class_ids': [0, 1, 2],
            'target_class_id': -1,
            'max_local_frame_age_sec': 0.80,
            'publish_debug_image': True, 'debug_image_fps': 5.0,
        }
        self.settings = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
            self.settings[name] = self.get_parameter(name).value
        cfg = self.settings
        for name in ('infer_fps', 'max_local_frame_age_sec', 'debug_image_fps'):
            if not math.isfinite(cfg[name]) or cfg[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < cfg['conf_threshold'] <= 1:
            raise ValueError('conf_threshold must be in (0, 1]')
        if cfg['model_task'] not in ('segment', 'detect'):
            raise ValueError('model_task must match the saved model: segment or detect')
        for name in ('expected_width', 'expected_height', 'imgsz', 'required_frames'):
            if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if (not cfg['allowed_class_ids'] or any(i not in CLASS_IDS.values()
                                               for i in cfg['allowed_class_ids'])):
            raise ValueError('allowed_class_ids must use the project IDs 0..5')
        if cfg['target_class_id'] != -1 and cfg['target_class_id'] not in cfg['allowed_class_ids']:
            raise ValueError('target_class_id must be -1 or included in allowed_class_ids')
        if (not 0 <= cfg['align_reference_x'] < cfg['expected_width']
                or not 0 < cfg['approach_stop_lower_y'] <= cfg['expected_height']):
            raise ValueError('Calibration coordinates must lie within the expected image')
        model_path = resolve_model_path(cfg['model_path'])
        self.model = YOLO(str(model_path), task=cfg['model_task'])
        # Exported Ultralytics models expose class metadata through model.names.
        # Validate the stable project class-name contract once at startup, then
        # use integer IDs only during inference. Class order may change and extra
        # model classes are allowed; only allowed_class_ids are required/used.
        model_names = self.model.names
        self._model_to_project_id = build_model_to_project_class_map(
            model_names, cfg['allowed_class_ids']
        )
        self.get_logger().info(
            f'MODEL_CONTRACT_OK: model_names={model_names}, '
            f'model_to_project_id={self._model_to_project_id}'
        )
        self._lock = Lock()
        self._frame = None
        self._frame_sequence = 0
        self._processed_sequence = 0
        self._tracking = False
        self._mode_epoch = 0
        self._selector = StableSelector(
            cfg['required_frames'], cfg['align_reference_x'],
            cfg['allowed_class_ids'], cfg['target_class_id'],
        )
        self._last_output_at = None
        self._last_debug_at = -math.inf
        self._last_error_at = -math.inf
        self._last_stats_at = time.monotonic()
        self._outputs = 0
        self._stats_outputs = 0
        self._camera_group = MutuallyExclusiveCallbackGroup()
        self._inference_group = MutuallyExclusiveCallbackGroup()
        self._service_group = MutuallyExclusiveCallbackGroup()
        camera_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subscription = self.create_subscription(
            CompressedImage, cfg['image_topic'], self.image_callback,
            camera_qos, callback_group=self._camera_group,
        )
        # Reliable result stream; depth 1 bounds middleware backlog at readers.
        self.publisher_ = self.create_publisher(DetectedObject, '/classified_detected_object_info', 1)
        self.image_pub = self.create_publisher(CompressedImage, '/yolo/image/compressed', camera_qos)
        self._service = self.create_service(
            SetTracking, 'set_tracking_mode', self.tracking_callback,
            callback_group=self._service_group,
        )
        self._timer = self.create_timer(
            1.0 / cfg['infer_fps'], self.infer_latest,
            callback_group=self._inference_group,
        )
        self.get_logger().info(
            f'YOLO_ONLY: no pre-classifier. model={model_path}, topic={cfg["image_topic"]}, '
            f'expected={cfg["expected_width"]}x{cfg["expected_height"]}; '
            'wait for YOLO_ONLY_READY before sending an Action goal.')

    def image_callback(self, msg):
        # Do not decode or infer here. Even during a slow inference, newer
        # callbacks replace this slot instead of queuing ten old images.
        with self._lock:
            self._frame_sequence += 1
            self._frame = (self._frame_sequence, time.monotonic(), msg)

    def tracking_callback(self, request, response):
        # Keep service failures machine-readable. AutoNav can distinguish a
        # temporarily cold/stale vision stream from a real service/config error.
        response.success = False
        response.reason = 'INTERNAL_ERROR'
        try:
            with self._lock:
                requested_class = int(request.target_class_id)
                if request.enable and requested_class not in self.settings['allowed_class_ids']:
                    response.reason = 'INVALID_TARGET_CLASS'
                    self.get_logger().warn(
                        f'추적 모드 거절: target_class_id={requested_class}가 허용 클래스가 아님')
                    return response
                if request.enable and (self._last_output_at is None
                                       or time.monotonic() - self._last_output_at > 2.0):
                    response.reason = 'VISION_NOT_READY'
                    self.get_logger().warn('추적 모드 거절: YOLO 워밍업/영상 수신 상태를 먼저 확인')
                    return response
                self._tracking = bool(request.enable)
                # Patrol/general mode keeps the YAML default (-1 = all allowed classes).
                # Tracking mode is locked to the class selected by AutoNav and carried
                # through RecycleActionMsg.index -> SetTracking.target_class_id.
                self._selector.target_class_id = (
                    requested_class if request.enable else int(self.settings['target_class_id'])
                )
                self._mode_epoch += 1
                self._selector.reset()
                # Drop an in-flight prediction produced under the previous mode.
            if request.enable:
                self.get_logger().info(f'추적 모드 ON: class_id={requested_class} 고정')
            else:
                self.get_logger().info('추적 모드 OFF: 전체 허용 클래스로 복귀')
            response.success = True
            response.reason = 'OK'
            return response
        except Exception as exc:
            self.get_logger().error(f'SetTracking 처리 예외: {type(exc).__name__}: {exc}')
            response.reason = 'INTERNAL_ERROR'
            return response

    def _warn(self, text):
        now = time.monotonic()
        if now - self._last_error_at >= 2.0:
            self.get_logger().warn(text)
            self._last_error_at = now

    def _candidates(self, result):
        boxes = result.boxes
        if boxes is None:
            return []
        candidates = []
        for cls, conf, coord in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywh.tolist()):
            # The class-name contract was already validated at startup. Keep the
            # hot inference path integer-only; unknown added model classes are
            # intentionally ignored rather than becoming project objects.
            project_id = self._model_to_project_id.get(int(cls))
            if project_id is None:
                continue
            candidates.append(
                Candidate(project_id, float(conf), *(float(v) for v in coord))
            )
        return candidates

    def infer_latest(self):
        with self._lock:
            item = self._frame
            if item is None or item[0] <= self._processed_sequence:
                # No new camera frame => no result heartbeat, no fake id=-1.
                return
            sequence, received_at, image_msg = item
            self._processed_sequence = sequence
            epoch = self._mode_epoch
        cfg = self.settings
        age_limit = cfg['max_local_frame_age_sec']
        if time.monotonic() - received_at > age_limit:
            self._warn('YOLO_ONLY: 오래된 입력 프레임 폐기')
            return
        try:
            frame = cv2.imdecode(np.frombuffer(image_msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('JPEG decode failed')
            if frame.shape[:2] != (cfg['expected_height'], cfg['expected_width']):
                raise ValueError(f'Image is {frame.shape[1]}x{frame.shape[0]}, expected '
                                 f'{cfg["expected_width"]}x{cfg["expected_height"]}; '
                                 'do not change calibrated pixel coordinates silently')
            results = self.model.predict(source=frame, imgsz=cfg['imgsz'],
                                         conf=cfg['conf_threshold'], verbose=False)
            if not results:
                raise ValueError('YOLO returned no Results object')
            candidates = self._candidates(results[0])
            finished_at = time.monotonic()
            local_age = finished_at - received_at
            if local_age > age_limit:
                # A slow/hung model is NOT a new, trustworthy no-object result.
                with self._lock:
                    self._selector.reset()
                self._warn(f'YOLO_ONLY: 처리 결과 지연 {local_age:.2f}s > {age_limit:.2f}s; 폐기')
                return
            with self._lock:
                if epoch != self._mode_epoch:
                    return
                selected = self._selector.select(candidates, self._tracking)
                output = DetectedObject()
                output.id = -1
                output.confidence = 0.0
                output.coord = [0.0, 0.0, 0.0, 0.0]
                output.min_y = 0.0
                if selected is not None:
                    output.id = selected.class_id
                    output.confidence = selected.confidence
                    output.coord = selected.coord
                    output.min_y = selected.y - selected.height / 2.0
                self.publisher_.publish(output)
                self._last_output_at = finished_at
                self._outputs += 1
            if self._outputs == 1:
                self.get_logger().info('YOLO_ONLY_READY: actual inference result stream started')
            if finished_at - self._last_stats_at >= 5.0:
                hz = (self._outputs - self._stats_outputs) / (finished_at - self._last_stats_at)
                self.get_logger().info(f'YOLO_ONLY output={hz:.2f}Hz, local_frame_age={local_age:.3f}s')
                self._last_stats_at, self._stats_outputs = finished_at, self._outputs
            self._publish_debug(frame, selected, image_msg)
        except Exception as exc:
            with self._lock:
                self._selector.reset()
            self._warn(f'YOLO_ONLY inference/input error: {type(exc).__name__}: {exc}; no result published')

    def _publish_debug(self, frame, selected, source_msg):
        cfg = self.settings
        now = time.monotonic()
        if not cfg['publish_debug_image'] or now - self._last_debug_at < 1.0 / cfg['debug_image_fps']:
            return
        self._last_debug_at = now
        # Visualization reuses the single inference result; no second YOLO.
        xref, ystop = int(cfg['align_reference_x']), int(cfg['approach_stop_lower_y'])
        cv2.line(frame, (xref, 0), (xref, frame.shape[0]-1), (0, 220, 255), 1)
        cv2.line(frame, (0, ystop), (frame.shape[1]-1, ystop), (0, 220, 255), 1)
        if selected is not None:
            x, y, w, h = selected.coord
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)),
                          (int(x+w/2), int(y+h/2)), (0, 220, 0), 2)
            text = f'id={selected.class_id} conf={selected.confidence:.2f} x={x:.0f} bottom={y+h/2:.0f}'
        else:
            text = 'no stable target'
        cv2.putText(frame, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            msg = CompressedImage()
            msg.header = source_msg.header
            msg.format = 'jpeg'
            msg.data = encoded.tobytes()
            self.image_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = executor = None
    try:
        node = YoloOnlyNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
