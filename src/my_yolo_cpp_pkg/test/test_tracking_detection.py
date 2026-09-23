"""모델과 ROS 통신 없이 최초 탐색 및 추적 중 검출 발행을 검증한다."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2

import numpy as np

import pytest

from std_msgs.msg import Header, Int32
from builtin_interfaces.msg import Time


class Boxes:
    """검출 결과에 필요한 좌표, 신뢰도, 클래스 배열을 제공한다."""

    def __init__(self, present=True, class_id=0):
        """한 물체의 검출 또는 빈 검출 결과를 만든다."""
        self.xywh = np.array(
            [[340.0, 300.0, 60.0, 80.0]] if present else []).reshape(-1, 4)
        self.conf = np.array([0.9] if present else [])
        self.cls = np.array([class_id] if present else [])

    def __len__(self):
        """검출된 물체 수를 반환한다."""
        return len(self.conf)


@pytest.fixture
def detector(monkeypatch):
    """실제 노드를 초기화하되 모델 추론과 ROS 입출력은 대체한다."""
    source = Path(__file__).parents[1] / 'my_yolo_cpp_pkg' / 'classify_yolo_information.py'
    spec = importlib.util.spec_from_file_location('tracking_detection_under_test', source)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=Mock()))
        imports.setitem(sys.modules, 'openvino', SimpleNamespace(Core=Mock()))
        spec.loader.exec_module(module)

    logger = Mock()
    publisher = Mock()
    ros_clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(module.Node, 'get_clock', lambda self: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(
            sec=int(ros_clock.now), nanosec=int((ros_clock.now % 1) * 1e9)))))
    monkeypatch.setattr(module.Node, '__init__', lambda self, name: None)
    monkeypatch.setattr(module.Node, 'declare_parameter',
                        lambda self, name, default: SimpleNamespace(value=default))
    monkeypatch.setattr(module.Node, 'create_subscription', lambda *a: None)
    monkeypatch.setattr(module.Node, 'destroy_subscription', lambda *a: None)
    monkeypatch.setattr(module.Node, 'create_timer', lambda *a: None)
    monkeypatch.setattr(module.Node, 'create_publisher', lambda *a: publisher)
    monkeypatch.setattr(module.Node, 'create_service', lambda *a: None)
    monkeypatch.setattr(module.Node, 'get_logger', lambda self: logger)
    monkeypatch.setattr(module.Node, 'get_parameter', lambda *a: SimpleNamespace(
        get_parameter_value=lambda: SimpleNamespace(double_value=0.5)))

    node = module.YoloNode()
    node.publisher_ = Mock()
    node.class_observation_pub = Mock()
    publish_image = node.publish_image
    node.publish_image = Mock()
    node.output_layer = 'output'
    node.compiled_classify_model = Mock(return_value={'output': np.array([[0.0, 1.0]])})
    node.model = Mock()
    node.model.predict.return_value = [SimpleNamespace(boxes=Boxes(), names={0: 'can'})]
    success, encoded = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
    assert success
    frame = SimpleNamespace(data=encoded.tobytes(), header=Header(frame_id='camera'))

    def observe():
        node.listener_callback(frame)
        return node.publisher_.publish.call_args.args[0]

    def set_tracking(enabled):
        response = node.srv_callback(
            SimpleNamespace(enable=enabled), SimpleNamespace(success=False))
        assert response.success

    return SimpleNamespace(node=node, observe=observe, set_tracking=set_tracking,
                           module=module, frame=frame, publish_image=publish_image, ros_clock=ros_clock)


def test_search_requires_two_frames_and_background_resets_confirmation(detector):
    """최초 탐색은 사전 분류와 연속 두 프레임 확인을 유지한다."""
    assert detector.observe().id == -1
    assert detector.observe().id == 0
    node = detector.node
    node.compiled_classify_model.return_value = {'output': np.array([[1.0, 0.0]])}
    node.last_detection_log_time = 0.0
    assert detector.observe().id == -1
    assert node.model.predict.call_count == 2

    node.compiled_classify_model.return_value = {'output': np.array([[0.0, 1.0]])}
    assert detector.observe().id == -1
    assert detector.observe().id == 0
    assert node.compiled_classify_model.call_count == 5


def test_tracking_bypasses_background_classifier_and_initial_confirmation(detector):
    """직전 배경 판정과 관계없이 추적 첫 검출의 좌표를 바로 발행한다."""
    node = detector.node
    node.compiled_classify_model.return_value = {'output': np.array([[1.0, 0.0]])}
    assert detector.observe().id == -1
    detector.set_tracking(True)
    node.compiled_classify_model.reset_mock()
    node.compiled_classify_model.side_effect = AssertionError('추적 중 사전 분류 실행')
    node.last_detection_log_time = 0.0

    message = detector.observe()
    assert message.id == 0
    assert list(message.coord) == [340.0, 300.0, 60.0, 80.0]
    assert message.confidence == pytest.approx(0.9)
    assert message.min_y == 260.0
    node.compiled_classify_model.assert_not_called()
    assert node.publish_image.call_count == 2


@pytest.mark.parametrize('missing', ['empty', 'unregistered'])
def test_tracking_reports_loss_and_publishes_first_reacquisition(detector, missing):
    """미검출은 그대로 알리고 대상 재등장 첫 프레임부터 유효 좌표를 보낸다."""
    detector.set_tracking(True)
    assert detector.observe().id == 0
    node = detector.node
    node.model.predict.return_value = [SimpleNamespace(
        boxes=Boxes(present=missing != 'empty'), names={0: 'unregistered'})]
    lost = detector.observe()
    assert lost.id == -1
    assert lost.confidence == 0.0
    assert list(lost.coord) == [0.0, 0.0, 0.0, 0.0]
    assert lost.min_y == 0.0

    node.model.predict.return_value = [SimpleNamespace(boxes=Boxes(), names={0: 'can'})]
    assert detector.observe().id == 0
    node.compiled_classify_model.assert_not_called()


def test_leaving_tracking_starts_a_new_two_frame_confirmation(detector):
    """추적 전의 확인 횟수가 남아 있어도 다음 탐색은 두 프레임을 새로 확인한다."""
    assert detector.observe().id == -1
    assert detector.observe().id == 0
    detector.set_tracking(True)
    assert detector.observe().id == 0
    detector.set_tracking(False)
    assert detector.observe().id == -1
    assert detector.observe().id == 0
    assert detector.node.compiled_classify_model.call_count == 4


def test_class_observation_preserves_flips_and_capture_stamp_before_confirmation(detector):
    """분류 투표에는 PENDING 프레임도 전달하고 원본 촬영 시각을 유지한다."""
    node = detector.node
    node.required_frames = 3
    for frame_number, name in enumerate(['trash', 'paper', 'trash', 'paper'], start=1):
        detector.frame.header.stamp.sec = frame_number
        node.model.predict.return_value = [SimpleNamespace(boxes=Boxes(), names={0: name})]
        assert detector.observe().id == -1
        batch = node.class_observation_pub.publish.call_args.args[0]
        assert batch.header.stamp.sec == 100
        observation = batch.detections[0]
        assert observation.header.stamp.sec == frame_number
        assert observation.results[0].hypothesis.class_id == name
        assert observation.results[0].hypothesis.score == pytest.approx(0.9)
        assert (observation.bbox.center.position.x, observation.bbox.center.position.y) == (340.0, 300.0)
        assert (observation.bbox.size_x, observation.bbox.size_y) == (60.0, 80.0)
    assert node.class_observation_pub.publish.call_count == node.model.predict.call_count == 4


def test_observation_uses_pc_time_before_inference_and_preserves_camera_time(detector):
    detector.frame.header.stamp.sec = 900
    detector.ros_clock.now = 100.0

    def slow_predict(**kwargs):
        detector.ros_clock.now = 102.0
        return [SimpleNamespace(boxes=Boxes(), names={0: 'can'})]

    detector.node.model.predict.side_effect = slow_predict
    detector.observe()
    batch = detector.node.class_observation_pub.publish.call_args.args[0]
    assert batch.header.stamp.sec == 100
    assert batch.detections[0].header.stamp.sec == 900
    assert detector.frame.header.stamp.sec == 900
    assert batch.header.frame_id == 'camera'


@pytest.mark.parametrize('missing', ['background', 'empty', 'unknown', 'paused'])
def test_missing_detection_does_not_repeat_class_observation(detector, missing):
    node = detector.node
    detector.observe()
    node.class_observation_pub.reset_mock()
    if missing == 'background':
        node.compiled_classify_model.return_value = {'output': np.array([[1.0, 0.0]])}
    elif missing == 'paused':
        node.inference_enabled = False
    else:
        node.model.predict.return_value = [SimpleNamespace(
            boxes=Boxes(present=missing != 'empty'), names={0: 'unknown'})]
    detector.observe()
    node.class_observation_pub.publish.assert_not_called()


@pytest.mark.parametrize('tracking', [False, True])
def test_repeated_mode_request_preserves_detection_progress(detector, tracking):
    """같은 모드 요청이 반복되어도 탐색 확인이나 추적 결과를 끊지 않는다."""
    detector.set_tracking(tracking)
    assert detector.observe().id == (0 if tracking else -1)
    detector.set_tracking(tracking)
    assert detector.observe().id == 0


@pytest.fixture
def visual_detector(detector, monkeypatch):
    """실제 그리기·JPEG 압축을 실행하고 영상과 제어 정보의 발행 순서를 기록한다."""
    node = detector.node
    node.publish_image = detector.publish_image
    node.image_pub = Mock()
    detector.clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(detector.module, 'time', SimpleNamespace(
        monotonic=lambda: detector.clock.now, time=lambda: 1000.0))
    detector.events = []
    node.publisher_.publish.side_effect = lambda msg: detector.events.append('control')
    node.image_pub.publish.side_effect = lambda msg: detector.events.append('image')
    detector.draw_text = Mock(wraps=cv2.putText)
    detector.draw_box = Mock(wraps=cv2.rectangle)
    monkeypatch.setattr(cv2, 'putText', detector.draw_text)
    monkeypatch.setattr(cv2, 'rectangle', detector.draw_box)
    return detector


def detection_boxes(rig):
    """글자 배경 사각형을 제외한 물체 테두리만 반환한다."""
    return [call for call in rig.draw_box.call_args_list if call.args[4] != cv2.FILLED]


def test_shared_image_marks_pending_then_accepted_target(visual_detector):
    """확인 중인 후보와 제어에 전달한 대상을 다른 색으로 같은 영상에 표시한다."""
    rig = visual_detector
    assert rig.observe().id == -1
    assert detection_boxes(rig)[-1].args[3] == (0, 255, 255)
    assert 'PENDING can 0.90' in [call.args[1] for call in rig.draw_text.call_args_list]

    rig.clock.now = 0.21
    rig.frame.header.stamp.sec = 42
    assert rig.observe().id == 0
    assert detection_boxes(rig)[-1].args[3] == (0, 255, 0)
    assert 'TARGET can 0.90' in [call.args[1] for call in rig.draw_text.call_args_list]
    image_msg = rig.node.image_pub.publish.call_args.args[0]
    assert image_msg.header == rig.frame.header
    assert image_msg.format == 'jpeg'
    decoded = cv2.imdecode(np.frombuffer(image_msg.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (480, 640, 3)
    assert decoded.any()
    assert rig.events == ['control', 'image', 'control', 'image']
    assert rig.node.model.predict.call_count == 2
    assert 'DETECTED: CAN 0.90 | SELECTED: -' in [
        call.args[1] for call in rig.draw_text.call_args_list]


def test_tracking_overlay_uses_control_target_instead_of_highest_confidence(visual_detector):
    """검출이 겹쳐도 제어 대상의 박스와 라벨만 하나씩 표시한다."""
    rig = visual_detector
    boxes = Boxes()
    boxes.xywh = np.array([[330.0, 300.0, 60.0, 80.0], [320.0, 300.0, 40.0, 80.0]])
    boxes.conf = np.array([0.95, 0.6])
    boxes.cls = np.array([0, 1])
    rig.node.model.predict.return_value = [SimpleNamespace(
        boxes=boxes, names={0: 'can', 1: 'paper'})]
    rig.set_tracking(True)
    message = rig.observe()
    assert message.id == 1
    assert message.coord[0] == 320.0
    colors = [call.args[3] for call in detection_boxes(rig)]
    assert colors == [(0, 255, 0)]
    labels = [call.args[1] for call in rig.draw_text.call_args_list]
    assert 'can 0.95' not in labels
    assert 'TARGET paper 0.60' in labels
    box_labels = [call.args[1] for call in rig.draw_text.call_args_list
                  if call.args[4] == 0.5 and call.args[5] == (0, 255, 255)]
    assert box_labels == ['TARGET paper 0.60']
    assert not any(label.startswith('TARGET can') for label in labels)
    assert 'DETECTED: PAPER 0.60 | SELECTED: -' in labels
    rig.node.compiled_classify_model.assert_not_called()
    assert rig.node.model.predict.call_count == 1


def test_image_rate_limit_does_not_skip_control_inference(visual_detector):
    """화면 발행을 제한해도 모든 입력 프레임에 대한 추론·제어 정보는 전달한다."""
    rig = visual_detector
    rig.set_tracking(True)
    for now in (0.0, 0.05, 0.10, 0.15, 0.21):
        rig.clock.now = now
        assert rig.observe().id == 0
    assert rig.node.model.predict.call_count == 5
    assert rig.node.publisher_.publish.call_count == 5
    assert rig.node.image_pub.publish.call_count == 2
    assert len(detection_boxes(rig)) == 2


@pytest.mark.parametrize('kind', ['background', 'empty', 'unregistered'])
def test_shared_image_does_not_reuse_previous_target(visual_detector, kind):
    """배경·미검출·미지원 클래스 화면에 이전 프레임의 TARGET을 남기지 않는다."""
    rig = visual_detector
    rig.observe()
    rig.clock.now = 0.21
    assert rig.observe().id == 0
    rig.draw_box.reset_mock()
    rig.draw_text.reset_mock()
    rig.clock.now = 0.42
    if kind == 'background':
        rig.node.compiled_classify_model.return_value = {'output': np.array([[1.0, 0.0]])}
    else:
        rig.node.model.predict.return_value = [SimpleNamespace(
            boxes=Boxes(present=kind != 'empty'), names={0: 'unregistered'})]
    assert rig.observe().id == -1
    assert rig.node.image_pub.publish.call_count == 3
    labels = [call.args[1] for call in rig.draw_text.call_args_list]
    assert not any('TARGET' in label for label in labels)
    if kind == 'unregistered':
        assert 'UNSUPPORTED unregistered 0.90' in labels
    else:
        assert not detection_boxes(rig)


def test_image_rendering_error_does_not_interrupt_control(visual_detector):
    """화면 그리기 오류가 발생한 다음 프레임도 정상적으로 검출 정보를 발행한다."""
    rig = visual_detector
    rig.node._draw_detections = Mock(side_effect=cv2.error('rendering failed'))
    rig.set_tracking(True)
    assert rig.observe().id == 0
    rig.clock.now = 0.21
    assert rig.observe().id == 0
    assert rig.node.publisher_.publish.call_count == 2
    rig.node.image_pub.publish.assert_not_called()


def set_inference(rig, enabled):
    return rig.node.inference_callback(
        SimpleNamespace(data=enabled), SimpleNamespace(success=False, message=''))


def test_pause_skips_both_models_and_only_decodes_low_rate_preview(visual_detector, monkeypatch):
    rig = visual_detector
    # 기본값과 관계없이 설정한 1 Hz 제한을 검증한다.
    rig.node.paused_image_hz = 1.0
    rig.observe()
    rig.clock.now = 0.21
    rig.observe()
    assert set_inference(rig, False).success
    assert rig.node.publisher_.publish.call_args.args[0].id == -1
    rig.node.publisher_.reset_mock()
    rig.node.model.predict.reset_mock()
    rig.node.compiled_classify_model.reset_mock()
    rig.node.image_pub.reset_mock()
    rig.draw_box.reset_mock()
    rig.draw_text.reset_mock()
    decode = Mock(wraps=cv2.imdecode)
    monkeypatch.setattr(cv2, 'imdecode', decode)
    for now in (1.0, 1.1, 1.4, 1.9, 2.01):
        rig.clock.now = now
        rig.node.listener_callback(rig.frame)
    rig.node.model.predict.assert_not_called()
    rig.node.compiled_classify_model.assert_not_called()
    rig.node.publisher_.publish.assert_not_called()
    assert decode.call_count == 2
    assert rig.node.image_pub.publish.call_count == 2
    labels = [call.args[1] for call in rig.draw_text.call_args_list if call.args[5] != (0, 0, 0)]
    assert labels.count('PAUSED | Inference off') == 2
    assert labels.count('DETECTED: - | SELECTED: -') == 2
    assert not detection_boxes(rig)


def test_paper_trash_difference_uses_navigation_choice_and_current_frame(visual_detector):
    """클래스가 달라도 글자 색은 유지하고 CLASS DIFF 문구로 차이를 알린다."""
    rig = visual_detector
    rig.set_tracking(True)
    rig.node.selected_class_callback(Int32(data=3))
    rig.node.model.predict.return_value = [SimpleNamespace(
        boxes=Boxes(class_id=1), names={1: 'paper', 3: 'trash'})]
    assert rig.observe().id == 1
    comparison = rig.draw_text.call_args
    assert comparison.args[1] == 'DETECTED: PAPER 0.90 | SELECTED: TRASH'
    assert comparison.args[2] == (10, 49)
    assert comparison.args[4] == 0.6
    assert comparison.args[5] == (0, 255, 255)
    assert any('CLASS DIFF' in call.args[1] for call in rig.draw_text.call_args_list)
    assert all(call.kwargs['lineType'] == cv2.LINE_AA
               for call in rig.draw_text.call_args_list)
    assert all(call.kwargs['lineType'] == cv2.LINE_AA for call in detection_boxes(rig))
    target_label = next(call for call in rig.draw_text.call_args_list
                        if call.args[1].startswith('TARGET '))
    assert target_label.args[4] == 0.5

    rig.clock.now = 0.21
    rig.draw_text.reset_mock()
    rig.node.model.predict.return_value[0].boxes = Boxes(class_id=3)
    assert rig.observe().id == 3
    comparison = rig.draw_text.call_args
    assert comparison.args[1] == 'DETECTED: TRASH 0.90 | SELECTED: TRASH'
    assert comparison.args[5] == (0, 255, 255)
    assert not any('CLASS DIFF' in call.args[1] for call in rig.draw_text.call_args_list)
    assert rig.node.selected_class_id == 3


def test_missing_detection_does_not_reuse_prediction_or_clear_adopted_class(visual_detector):
    rig = visual_detector
    rig.node.selected_class_callback(Int32(data=1))
    rig.observe()
    rig.clock.now = 0.21
    rig.node.model.predict.return_value = [SimpleNamespace(boxes=Boxes(False), names={0: 'can'})]
    assert rig.observe().id == -1
    assert rig.draw_text.call_args.args[1] == 'DETECTED: - | SELECTED: PAPER'

    rig.node.selected_class_callback(Int32(data=-1))
    rig.clock.now = 0.42
    rig.observe()
    assert rig.draw_text.call_args.args[1] == 'DETECTED: - | SELECTED: -'


def test_paused_preview_keeps_adopted_class_until_navigation_clears_it(visual_detector):
    rig = visual_detector
    rig.node.selected_class_callback(Int32(data=3))
    assert set_inference(rig, False).success
    rig.node.listener_callback(rig.frame)
    assert rig.draw_text.call_args.args[1] == 'DETECTED: - | SELECTED: TRASH'
    rig.node.model.predict.assert_not_called()
    rig.node.compiled_classify_model.assert_not_called()
    rig.node.selected_class_callback(Int32(data=-1))
    rig.clock.now = 1.1
    rig.node.listener_callback(rig.frame)
    assert rig.draw_text.call_args.args[1] == 'DETECTED: - | SELECTED: -'


def test_box_label_stays_inside_frame_near_right_edge(visual_detector):
    rig = visual_detector
    boxes = Boxes(class_id=2)
    boxes.xywh = np.array([[620.0, 100.0, 35.0, 70.0]])
    rig.node.model.predict.return_value = [SimpleNamespace(boxes=boxes, names={2: 'plastic'})]
    rig.set_tracking(True)
    rig.observe()
    label = next(call for call in rig.draw_text.call_args_list
                 if call.args[1].startswith('TARGET '))
    (_, text, (x, y), font, scale, _, thickness) = label.args
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    assert 0 <= x and x + width < 640
    assert y - height > 60


@pytest.mark.parametrize('value', [0, 127, 255])
def test_overlay_keeps_background_and_uses_fixed_yellow_outlined_text(detector, monkeypatch, value):
    frame = np.full((480, 640, 3), value, dtype=np.uint8)
    text = Mock(wraps=cv2.putText)
    rectangle = Mock(wraps=cv2.rectangle)
    monkeypatch.setattr(cv2, 'putText', text)
    monkeypatch.setattr(cv2, 'rectangle', rectangle)
    detector.node.selected_class_id = 3
    detector.node._draw_detections(
        frame, SimpleNamespace(boxes=Boxes(), names={0: 'can'}), 0, True)
    # 상단 바와 라벨 바탕 모두 사라지고 글자 바깥의 원본 영상은 유지된다.
    assert all(call.args[4] != cv2.FILLED for call in rectangle.call_args_list)
    assert np.all(frame[0:60, 600:] == value)
    colors = [call.args[5] for call in text.call_args_list]
    assert colors == [(0, 0, 0), (0, 255, 255)] * 3
    assert [call.args[4] for call in text.call_args_list] == [0.5, 0.5, 0.6, 0.6, 0.6, 0.6]
    assert np.any(np.all(frame[:60] == (0, 255, 255), axis=2))


def test_resume_clears_confirmation_and_periodic_on_preserves_new_progress(detector):
    rig = detector
    rig.observe()
    assert rig.observe().id == 0
    assert set_inference(rig, False).success
    assert set_inference(rig, True).success
    assert rig.observe().id == -1
    assert set_inference(rig, True).success
    assert rig.observe().id == 0


def test_pause_watchdog_recovers_without_camera_or_navigation(visual_detector):
    rig = visual_detector
    assert set_inference(rig, False).success
    rig.clock.now = 4.0
    assert set_inference(rig, False).success
    rig.clock.now = 5.1
    rig.node._check_pause_timeout()
    assert not rig.node.inference_enabled
    rig.clock.now = 9.1
    rig.node._check_pause_timeout()
    assert rig.node.inference_enabled
    assert rig.observe().id == -1
    assert rig.node.model.predict.call_count == 1


def test_tracking_wakes_inference_and_rejects_pause_until_tracking_finishes(detector):
    rig = detector
    assert set_inference(rig, False).success
    rig.set_tracking(True)
    assert rig.node.inference_enabled
    assert not set_inference(rig, False).success
    assert rig.observe().id == 0
    rig.set_tracking(False)
    assert set_inference(rig, False).success


def test_mode_transition_discards_queued_camera_callbacks(detector, monkeypatch):
    rig = detector
    callbacks = []
    monkeypatch.setattr(rig.module.Node, 'create_subscription',
                        lambda self, kind, topic, callback, qos: callbacks.append(callback))
    rig.node._subscribe_images()
    old_callback = callbacks[-1]
    set_inference(rig, False)
    paused_callback = callbacks[-1]
    set_inference(rig, True)
    rig.node.publisher_.reset_mock()
    old_callback(rig.frame)
    paused_callback(rig.frame)
    rig.node.model.predict.assert_not_called()
    rig.node.publisher_.publish.assert_not_called()
    callbacks[-1](rig.frame)
    assert rig.node.publisher_.publish.call_args.args[0].id == -1
