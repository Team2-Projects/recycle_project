"""모델 없는 영상 뷰어의 구독, 창 갱신 및 종료 동작을 검증한다."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2

import numpy as np

import pytest


@pytest.fixture
def viewer_module(monkeypatch):
    """추론 라이브러리 없이 화면 모듈을 읽고 ROS 통신을 대체한다."""
    source = Path(__file__).parents[1] / 'my_yolo_cpp_pkg' / 'best_yolo_node.py'
    spec = importlib.util.spec_from_file_location('viewer_under_test', source)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, 'ultralytics', None)
        imports.setitem(sys.modules, 'openvino', None)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module.Node, '__init__', lambda self, name: None)
    monkeypatch.setattr(module.Node, 'create_subscription', Mock())
    monkeypatch.setattr(module.Node, 'destroy_node', Mock())
    monkeypatch.setattr(module.Node, 'get_logger', lambda self: Mock())
    return module


def test_viewer_uses_same_latest_image_topic_as_web(viewer_module):
    """뷰어는 제어용 노드가 발행하는 영상만 최신 하나씩 구독한다."""
    node = viewer_module.YoloImageViewer()
    subscription = node.create_subscription.call_args.args
    assert subscription[1] == '/yolo/image/compressed'
    qos = subscription[3]
    assert qos.reliability == viewer_module.ReliabilityPolicy.BEST_EFFORT
    assert qos.history == viewer_module.HistoryPolicy.KEEP_LAST
    assert qos.depth == 1


def test_viewer_decodes_latest_image_and_ignores_corrupt_data(viewer_module):
    """정상 영상은 갱신하고 비어 있거나 손상된 데이터는 화면을 덮어쓰지 않는다."""
    node = viewer_module.YoloImageViewer()
    for value in (30, 180):
        frame = np.full((48, 64, 3), value, dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', frame)
        assert success
        node.listener_callback(SimpleNamespace(data=encoded.tobytes()))
        np.testing.assert_array_equal(node.latest_frame, frame)
    latest = node.latest_frame
    for data in (b'', b'not a jpeg'):
        node.listener_callback(SimpleNamespace(data=data))
        assert node.latest_frame is latest


@pytest.mark.parametrize('exit_method', ['escape', 'q', 'close', 'interrupt', 'gui_error'])
def test_viewer_opens_without_images_and_cleans_up(viewer_module, monkeypatch, exit_method):
    """영상 대기 중에도 창을 열며 창 닫기나 종료 요청 시 자원을 해제한다."""
    module = viewer_module
    monkeypatch.setattr(module.rclpy, 'init', Mock())
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
    shutdown = Mock()
    monkeypatch.setattr(module.rclpy, 'shutdown', shutdown)
    spin = Mock(side_effect=KeyboardInterrupt if exit_method == 'interrupt' else None)
    monkeypatch.setattr(module.rclpy, 'spin_once', spin)
    opened, displayed, destroyed = Mock(), Mock(), Mock()
    monkeypatch.setattr(cv2, 'namedWindow', opened)
    monkeypatch.setattr(cv2, 'imshow', displayed)
    monkeypatch.setattr(cv2, 'destroyAllWindows', destroyed)
    key = {'escape': 27, 'q': ord('q')}.get(exit_method, -1)
    wait_key = Mock(return_value=key)
    if exit_method == 'gui_error':
        wait_key.side_effect = cv2.error('display unavailable')
    monkeypatch.setattr(cv2, 'waitKey', wait_key)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 0)

    module.main()

    opened.assert_called_once_with(module.WINDOW_NAME, cv2.WINDOW_NORMAL)
    displayed.assert_called_once()
    destroyed.assert_called_once()
    module.Node.destroy_node.assert_called_once()
    shutdown.assert_called_once()


def test_viewer_displays_received_frame_without_inference(viewer_module, monkeypatch):
    """ROS에서 받은 프레임을 표시하고 종료하며 추가 추론을 요구하지 않는다."""
    module = viewer_module
    frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    monkeypatch.setattr(module.rclpy, 'init', Mock())
    monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(module.rclpy, 'shutdown', Mock())
    monkeypatch.setattr(module.rclpy, 'spin_once',
                        lambda node, **kwargs: setattr(node, 'latest_frame', frame))
    displayed = Mock()
    monkeypatch.setattr(cv2, 'namedWindow', Mock())
    monkeypatch.setattr(cv2, 'imshow', displayed)
    monkeypatch.setattr(cv2, 'destroyAllWindows', Mock())
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: 27)

    module.main()

    assert displayed.call_count == 2
    assert displayed.call_args.args[1] is frame
