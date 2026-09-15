"""Exercise the real bridge subscription and WebSocket formatting offline."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


@pytest.fixture
def bridge(monkeypatch):
    class Socket:
        def __init__(self):
            self.messages = []

        def connect(self, url):
            self.url = url

        def send(self, payload, **kwargs):
            self.messages.append(json.loads(payload))

    class Node:
        def __init__(self, name):
            self.subscriptions, self.logs = {}, []

        def create_subscription(self, message_type, topic, callback, qos):
            self.subscriptions[topic] = callback
            return NS()

        def create_publisher(self, *args):
            return NS(publish=lambda msg: None)

        def create_timer(self, *args):
            return NS()

        def get_logger(self):
            return NS(error=self.logs.append, warn=self.logs.append)

    modules = {
        'rclpy': {}, 'rclpy.node': dict(Node=Node),
        'rclpy.qos': dict(QoSProfile=lambda **kw: NS(**kw),
                           ReliabilityPolicy=NS(BEST_EFFORT=1),
                           HistoryPolicy=NS(KEEP_LAST=1), DurabilityPolicy=NS()),
        'sensor_msgs.msg': dict(BatteryState=object, CompressedImage=object),
        'std_msgs.msg': dict(String=lambda **kw: NS(**kw)),
        'geometry_msgs.msg': dict(PoseStamped=object),
        'nav_msgs.msg': dict(Path=object),
        'tf2_ros': dict(Buffer=lambda: NS(), TransformListener=lambda *args: NS()),
        'websocket': dict(WebSocket=Socket),
        'psutil': dict(cpu_percent=lambda **kw: 0.),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / 'ros_bridge/ros_spring_bridge.py'
    spec = importlib.util.spec_from_file_location('_recovery_bridge_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpringBridge()


def test_recovery_heartbeat_reaches_existing_status_channel_without_task_history(bridge):
    callback = bridge.subscriptions['/auto_nav/recovery_details']
    for sequence in (1, 2):
        details = dict(state='RECOVERY_WAIT_SCAN_STALE', sequence=sequence,
                       destination='HOME', reason='BATTERY_LOW',
                       elapsed_sec=30. + sequence, state_elapsed_sec=29. + sequence,
                       waiting=True, manual=False)
        callback(NS(data=json.dumps(details)))
        event = bridge.ws.messages[-1]
        assert event == dict(type='robot_status', eventType='recovery',
                             status=details['state'], recovery=details)
    assert len(bridge.ws.messages) == 2
    assert not bridge.should_shutdown


@pytest.mark.parametrize('payload', ['not json', 'null', '[]', '{}', '{"state": 12}'])
def test_invalid_telemetry_does_not_stop_bridge_or_publish_motion_state(bridge, payload):
    bridge.subscriptions['/auto_nav/recovery_details'](NS(data=payload))
    assert bridge.ws.messages == []
    assert not bridge.should_shutdown
