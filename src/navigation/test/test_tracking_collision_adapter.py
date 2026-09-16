"""Run the real action adapter with in-process ROS API doubles, NOT DDS tests."""
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.collision_safety import CollisionConfig, CollisionSafety  # noqa: E402
from navigation.tracking_control import (  # noqa: E402
    Command, Phase, TrackingConfig, TrackingController,
)


class Twist:
    def __init__(self):
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class PolygonStamped:
    def __init__(self):
        self.header = NS(stamp=NS(sec=0, nanosec=0), frame_id='')
        self.polygon = NS(points=[])


class Pub:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class Event:
    def __init__(self, harness):
        self.harness = harness
        self.value = False

    def set(self):
        self.value = True

    def clear(self):
        self.value = False

    def is_set(self):
        return self.value

    def wait(self, delay):
        if not self.value:
            self.harness.advance(delay)
        return self.value


class Goal:
    def __init__(self):
        self.request = NS(index=0)
        self.is_cancel_requested = False
        self.state = ''
        self.feedback = []

    def publish_feedback(self, feedback):
        self.feedback.append(feedback.status)

    def succeed(self):
        self.state = 'SUCCEEDED'

    def abort(self):
        self.state = 'ABORTED'

    def canceled(self):
        self.state = 'CANCELED'


@pytest.fixture
def runtime(monkeypatch):
    h = NS(now=0.0)

    class FakeNode:
        def __init__(self, name):
            self.name = name
            self.params = {}
            self.pubs = {}
            self.subs = {}
            self.sub_qos = {}
            self.timers = []
            self.logs = []
            self.tf_ok = True
            self.extra_safe = False

        def declare_parameter(self, name, default, *args):
            self.params.setdefault(name, default)

        def get_parameter(self, name):
            return NS(value=self.params[name])

        def create_publisher(self, cls, topic, qos):
            p = Pub(topic)
            self.pubs[topic] = p
            return p

        def create_subscription(self, cls, topic, cb, qos, **kwargs):
            self.subs[topic] = cb
            self.sub_qos[topic] = qos
            return NS()

        def create_client(self, *args, **kwargs):
            return NS()

        def create_timer(self, period, cb, **kwargs):
            self.timers.append((period, cb, kwargs))
            return NS()

        def get_name(self):
            return self.name

        def get_clock(self):
            def now():
                ns = int((1700000000 + h.now + getattr(h, 'ros_offset', 0.)) * 1e9)
                return NS(nanoseconds=ns,
                          to_msg=lambda: NS(sec=ns // 10**9, nanosec=ns % 10**9))
            return NS(now=now)

        def get_logger(self):
            return NS(info=self.logs.append, warn=self.logs.append, error=self.logs.append)

        def get_subscriptions_info_by_topic(self, topic):
            if topic == '/tracking_collision/scan' and not getattr(self, 'legacy_scan_route', False):
                return [NS(node_name='tracking_collision_monitor')]
            return []

        def get_publishers_info_by_topic(self, topic):
            if topic == '/cmd_vel':
                return [NS(node_name=name) for name in getattr(self, 'wheel_names',
                        ['recycle_tracking_node', 'auto_nav', 'controller_server', 'recycle'])]
            names = (['recycle_tracking_node']
                     if topic.endswith('_raw') or topic == '/tracking_collision/scan'
                     else ['tracking_collision_monitor'])
            if self.extra_safe and topic.endswith('_safe'):
                names.append('collision_monitor')
            return [NS(node_name=name) for name in names]

    class Buffer:
        def __init__(self):
            self.ok = True

        def can_transform(self, target, source, stamp, return_debug_tuple=False):
            good = self.ok and target == 'base_footprint' and source == 'base_scan'
            if return_debug_tuple:
                return (good, '' if good else 'mock: base_scan transform missing')
            return good

    modules = {
        'rclpy': {'ok': lambda: True},
        'rclpy.action': {'ActionServer': lambda *a, **kw: NS(),
                         'CancelResponse': NS(ACCEPT=1), 'GoalResponse': NS(ACCEPT=1, REJECT=0)},
        'rclpy.callback_groups': {'MutuallyExclusiveCallbackGroup': type('Group', (), {}),
                                  'ReentrantCallbackGroup': type('Group2', (), {})},
        'rclpy.executors': {'ExternalShutdownException': type('Shutdown', (Exception,), {}),
                            'MultiThreadedExecutor': object},
        'rclpy.node': {'Node': FakeNode},
        'rclpy.qos': {'QoSProfile': lambda **kw: NS(**kw),
                      'ReliabilityPolicy': NS(BEST_EFFORT=1),
                      'DurabilityPolicy': NS(VOLATILE=1)},
        'rclpy.clock': {'Clock': lambda **kw: NS(), 'ClockType': NS(STEADY_TIME=1)},
        'rclpy.time': {'Time': lambda: NS()},
        'tf2_ros': {'Buffer': Buffer, 'TransformListener': lambda *a: NS()},
        'sensor_msgs.msg': {'LaserScan': type('LaserScan', (), {})},
        'std_msgs.msg': {'String': lambda data='': NS(data=data)},
        'rcl_interfaces.msg': {'ParameterDescriptor': lambda **kw: NS(**kw)},
        'geometry_msgs.msg': {'Point32': lambda **kw: NS(**kw),
                              'PolygonStamped': PolygonStamped, 'Twist': Twist},
        'my_yolo_msgs.msg': {'DetectedObject': type('DetectedObject', (), {})},
        'my_yolo_msgs.srv': {'SetTracking': NS(Request=lambda: NS())},
        'navigation_interface.msg': {'CollisionCommand': lambda request_id=0, velocity=None, input_valid=False: NS(request_id=request_id, velocity=velocity or Twist(), input_valid=input_valid)},
        'navigation_interface.action': {'RecycleActionMsg': NS(
            Feedback=lambda: NS(status=''), Result=lambda: NS(success=False, message=''))},
    }
    for name, attrs in modules.items():
        m = ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
    path = Path(__file__).resolve().parents[1] / 'navigation/recycle_tracking_node.py'
    spec = importlib.util.spec_from_file_location('_tracking_collision_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time = NS(monotonic=lambda: h.now)
    n = mod.RecycleTrackingNode()
    h.node = n
    h.mod = mod
    h.goal = Goal()
    h.responses = []
    h.max_time = 40.0
    h.scan_on = True
    h.monitor_on = True
    h.vision_on = True
    h.ratio = lambda t, raw: 1.0
    h.position = lambda t: (350.0, min(435., 280. + max(0., t - 1) * 45.))
    h.last_vision = -10.0
    h.seq = 0
    h.cancel_at = None
    h.env_hook = lambda: None
    h.trace = []

    h.tf_ok = True
    def env():
        h.env_hook()

    def reply(raw, ratio=1., valid=True):
        return NS(request_id=raw.request_id, input_valid=valid,
                  velocity=n._twist(Command(raw.velocity.linear.x * ratio,
                                             raw.velocity.angular.z * ratio)))
    h.reply = reply

    def advance(dt):
        h.now = round(h.now + max(dt, .0001), 8)
        env()
        if h.cancel_at is not None and h.now >= h.cancel_at:
            h.goal.is_cancel_requested = True
            n.cancel_callback(h.goal)
        if h.monitor_on and n._raw_pub.messages:
            raw = n._raw_pub.messages[-1]
            r = h.ratio(h.now, raw.velocity)
            if r is not None:
                safe = h.reply(raw, r, h.scan_on and h.tf_ok)
                n._safe_callback(safe)
        if h.vision_on and h.now - h.last_vision >= .199:
            h.last_vision = h.now
            h.seq += 1
            x, bottom = h.position(h.now)
            n.obj_callback(NS(id=0, confidence=.9, coord=[x, bottom - 25., 40., 50.]))
        n._collision_watchdog()
        output = n.cmd_vel_pub.messages[-1] if n.cmd_vel_pub.messages else Twist()
        h.trace.append((h.now, n._last_collision_status, output.linear.x, output.angular.z,
                        n._controller.phase if n._controller else None))

    h.advance = advance
    h.environment = env
    n.cancel_event = Event(h)
    n._running = lambda: h.now < h.max_time
    n.call_tracking_srv = lambda *a, **kw: (True, 'OK')
    return h


def states(h):
    return [line.removeprefix('Collision: ') for line in h.node.logs if line.startswith('Collision: ')]


def test_idle_never_publishes_cmd_vel(runtime):
    h = runtime
    h.advance(.1)
    h.node._safe_callback(Twist())
    h.node._publish_command(Command(.08, 0))
    h.node.request_shutdown()
    assert not h.node.cmd_vel_pub.messages
    assert not h.node._raw_pub.messages


def test_ready_waits_for_monitor_and_geometry(runtime):
    h = runtime
    h.monitor_on = False
    result = h.node.execute_callback(h.goal)
    assert h.goal.state == 'ABORTED'
    assert result.message.startswith('SAFETY_NOT_READY:')
    assert all(m.linear.x == 0 and m.angular.z == 0 for m in h.node.cmd_vel_pub.messages)


@pytest.mark.parametrize('input_valid,monitor_on', [(False, True), (True, False)])
def test_no_valid_correlated_response_never_starts_motion(runtime, input_valid, monitor_on):
    h = runtime
    h.scan_on, h.monitor_on = input_valid, monitor_on
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('SAFETY_NOT_READY:')
    assert not result.success
    assert all(m.linear.x == 0 and m.angular.z == 0 for m in h.node.cmd_vel_pub.messages)


def test_raw_has_no_direct_motion_path(runtime):
    h = runtime
    n = h.node
    n._owns_cmd_vel = True
    n._publish_command(Command(.08, 0))
    assert n._raw_pub.messages[-1].velocity.linear.x == .08
    assert all(msg.linear.x == 0 for msg in n.cmd_vel_pub.messages)


def test_normal_test_mode_ends_and_releases_ownership(runtime):
    h = runtime
    result = h.node.execute_callback(h.goal)
    assert h.goal.state == 'ABORTED' and result.message.startswith('TEST_STOP:')
    assert any(m.linear.x > 0 for m in h.node.cmd_vel_pub.messages)
    assert not h.node._owns_cmd_vel
    count = len(h.node.cmd_vel_pub.messages)
    h.node._safe_callback(h.node._twist(Command(.08, 0)))
    h.node._collision_watchdog()
    h.node.request_shutdown()
    assert len(h.node.cmd_vel_pub.messages) == count


def test_normal_collection_succeeds_only_after_final_motion(runtime):
    h = runtime
    h.node.config = TrackingConfig(stop_only_test_mode=False, final_approach_calibrated=True)
    result = h.node.execute_callback(h.goal)
    assert result.success and h.goal.state == 'SUCCEEDED'
    moving_final = [row[0] for row in h.trace if row[2] == .03 and row[4] == Phase.FINAL_APPROACH]
    assert moving_final and h.now - moving_final[0] >= 10.0
    assert not h.node._owns_cmd_vel


def test_collision_hold_blocks_and_stops_hypothetical_probe(runtime):
    h = runtime
    h.position = lambda t: (350., 300.)
    h.ratio = lambda t, raw: 0.0 if t > 1 and raw.linear.x > 0 else 1.0
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('COLLISION_BLOCKED:')
    assert 'COLLISION_HOLD' in states(h)
    after_hold = False
    for t, state, linear, angular, phase in h.trace:
        if state == 'COLLISION_HOLD':
            after_hold = True
        if after_hold:
            assert linear == 0 and angular == 0
    assert not result.success


def test_boundary_jitter_keeps_deadline(runtime):
    h = runtime
    h.position = lambda t: (350., 300.)
    def ratio(t, raw):
        if t < 1 or raw.linear.x == 0:
            return 1.
        return [0., .10, 0., .15][int(t * 10) % 4]
    h.ratio = ratio
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('COLLISION_BLOCKED:')
    assert states(h).count('COLLISION_HOLD') <= 2


def test_hold_release_requires_actual_visual_realign(runtime):
    h = runtime
    h.ratio = lambda t, raw: 0. if 1.0 < t < 1.6 and raw.linear.x > 0 else 1.
    result = h.node.execute_callback(h.goal)
    assert 'REALIGN_REQUIRED' in states(h)
    assert result.message.startswith('TEST_STOP:')
    # Resume step uses new alignment observations, not immediately the old raw.
    realign_times = [t for t, state, x, z, p in h.trace if p == Phase.REALIGN]
    assert len(realign_times) >= 3


def test_near_observations_during_hold_cannot_start_final(runtime):
    h = runtime
    h.node.config = TrackingConfig(stop_only_test_mode=False, final_approach_calibrated=True)
    h.position = lambda t: (350., 300. if t < 1.2 else 435.)
    h.ratio = lambda t, raw: 0. if t > 1 and raw.linear.x > 0 else 1.
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('COLLISION_BLOCKED:')
    assert not any(row[4] == Phase.FINAL_APPROACH for row in h.trace)


def test_final_intervention_is_terminal(runtime):
    h = runtime
    h.node.config = TrackingConfig(stop_only_test_mode=False, final_approach_calibrated=True)
    h.ratio = lambda t, raw: .98 if raw.linear.x == .03 and t > 5.5 else 1.
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert not result.success
    assert 'FINAL_APPROACH_INTERRUPTED' in states(h)


@pytest.mark.parametrize('sensor', ['scan', 'safe', 'tf'])
def test_final_watchdog_interrupts(sensor, runtime):
    h = runtime
    h.node.config = TrackingConfig(stop_only_test_mode=False, final_approach_calibrated=True)
    def hook():
        if h.now > 6:
            if sensor == 'scan':
                h.scan_on = False
            elif sensor == 'safe':
                h.monitor_on = False
            else:
                h.tf_ok = False
    h.env_hook = hook
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert not result.success


def test_cancel_stops_then_late_safe_ignored(runtime):
    h = runtime
    h.cancel_at = 1.5
    result = h.node.execute_callback(h.goal)
    assert h.goal.state == 'CANCELED' and not result.success
    assert h.node.cmd_vel_pub.messages[-1].linear.x == 0
    count = len(h.node.cmd_vel_pub.messages)
    h.node._safe_callback(h.node._twist(Command(.08, 0)))
    assert len(h.node.cmd_vel_pub.messages) == count


def test_sensor_outage_stops_and_fails_bounded(runtime):
    h = runtime
    h.position = lambda t: (350., 300.)
    h.env_hook = lambda: setattr(h, 'scan_on', False) if h.now >= 1 else None
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('SAFETY_UNAVAILABLE:')
    assert h.now < 5


def test_cleanup_failure_preserves_collision_reason(runtime):
    h = runtime
    h.position = lambda t: (350., 300.)
    h.ratio = lambda t, raw: 0. if t > 1 and raw.linear.x > 0 else 1.
    h.node.call_tracking_srv = lambda enable, *a, **kw: (True, 'OK') if enable else (False, 'SERVICE_TIMEOUT')
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('COLLISION_BLOCKED:')
    assert 'CLEANUP_UNCONFIRMED' in result.message


def test_shutdown_while_active_zeros(runtime):
    h = runtime
    h.node._owns_cmd_vel = True
    h.node._controller = TrackingController(TrackingConfig(), 0, 0)
    h.node.request_shutdown()
    assert h.node.cmd_vel_pub.messages[-1].linear.x == 0
    assert h.node._raw_pub.messages[-1].velocity.linear.x == 0


def test_delayed_moving_reply_does_not_override_vision_brake(runtime):
    h, n = runtime, runtime.node
    n._owns_cmd_vel = True
    n._controller = TrackingController(TrackingConfig(), 0, 0)
    n._publish_command(Command(.08, 0))
    late = h.reply(n._raw_pub.messages[-1])
    n._publish_command()
    n._safe_callback(late)
    assert not n._safety.failure
    assert all(m.linear.x == 0 for m in n.cmd_vel_pub.messages)


def test_slowed_to_physical_stall_is_not_generic_patrol_retry(runtime):
    h = runtime
    h.position = lambda t: (350., 300.)
    h.ratio = lambda t, raw: .15 if t > 1 and raw.linear.x > 0 else 1.
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('COLLISION_BLOCKED:')
    assert 'COLLISION_SLOWING' in states(h)


def test_alignment_collision_and_clear(runtime):
    h = runtime
    h.position = lambda t: (300. if t < 2.5 else 350., min(435., 280. + max(0., t - 2.5) * 45.))
    h.ratio = lambda t, raw: 0. if .8 < t < 1.3 and raw.angular.z > 0 else 1.
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('TEST_STOP:')
    assert 'COLLISION_HOLD' in states(h)
    assert 'REALIGN_REQUIRED' in states(h)
    assert any(m.angular.z > 0 for m in h.node.cmd_vel_pub.messages)


def test_monitor_input_gap_retires_attempt_instead_of_recovering_in_place(runtime):
    h = runtime
    h.env_hook = lambda: setattr(h, 'scan_on', not (1.1 < h.now < 1.8))
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('SAFETY_UNAVAILABLE:')
    assert not h.node._owns_cmd_vel
    assert not result.success


def test_cancelled_early_never_moves(runtime):
    h = runtime
    h.node.cancel_event.set()
    h.goal.is_cancel_requested = True
    result = h.node.execute_callback(h.goal)
    assert not result.success and h.goal.state == 'CANCELED'
    assert all(m.linear.x == 0 and m.angular.z == 0 for m in h.node.cmd_vel_pub.messages)


def test_readiness_and_service_failure_never_get_control_motion(runtime):
    h = runtime
    h.node.call_tracking_srv = lambda *a, **kw: (False, 'VISION_NOT_READY')
    result = h.node.execute_callback(h.goal)
    assert not result.success
    assert not h.node._owns_cmd_vel
    assert all(m.linear.x == 0 and m.angular.z == 0 for m in h.node.cmd_vel_pub.messages)


def test_tracking_uses_no_graph_queries_or_scan_geometry_topics(runtime, monkeypatch):
    h = runtime
    def forbidden(*a, **kw):
        raise AssertionError('Tracking must not inspect ROS graph')
    monkeypatch.setattr(h.node, 'get_publishers_info_by_topic', forbidden)
    monkeypatch.setattr(h.node, 'get_subscriptions_info_by_topic', forbidden)
    assert '/scan' not in h.node.subs
    assert set(h.node.pubs) == {'/cmd_vel', '/tracking_cmd_vel_raw'}
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('TEST_STOP:'), result.message


def test_previous_action_response_cannot_authorize_new_identical_intent(runtime):
    h, n = runtime, runtime.node
    n._owns_cmd_vel = True
    n._controller = TrackingController(TrackingConfig(), 0, 0)
    n._publish_command(Command(.08, 0))
    old = h.reply(n._raw_pub.messages[-1])
    n._safety = CollisionSafety(CollisionConfig())
    n._publish_command(Command(.08, 0))
    n._safe_callback(old)
    assert all(m.linear.x == 0 for m in n.cmd_vel_pub.messages)
    n._safe_callback(h.reply(n._raw_pub.messages[-1]))
    assert n.cmd_vel_pub.messages[-1].linear.x == .08


def test_advancing_clock_between_function_calls_no_false_raw_stale(runtime):
    """This test FAILS on the original adapter despite its 89 passing tests."""
    h = runtime

    def progressing_clock():
        h.now += .0001  # represent time spent between real Python calls
        return h.now

    h.mod.time = NS(monotonic=progressing_clock)
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('TEST_STOP:'), result.message
    assert 'RAW_STALE' not in states(h)
    assert 'RAW_RECEIPT_CLOCK_ORDER' not in states(h)
    assert 'SAFETY_UNAVAILABLE' not in states(h)
    assert any(m.linear.x > 0 for m in h.node.cmd_vel_pub.messages)


def make_scan(h, offset=0., stamp_ns=None, frame='base_scan'):
    ns = (h.node.get_clock().now().nanoseconds + round(offset * 1e9)
          if stamp_ns is None else stamp_ns)
    return NS(header=NS(stamp=NS(sec=ns // 10**9, nanosec=ns % 10**9), frame_id=frame),
              angle_min=0., angle_max=6.28, angle_increment=.025,
              range_min=0., range_max=100., ranges=[2., 2., 3., 4.])


def test_tf_failure_retires_attempt_and_late_healthy_output_cannot_restart(runtime):
    h = runtime
    h.env_hook = lambda: setattr(h, 'tf_ok', not (1.3 < h.now < 1.6))
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('SAFETY_UNAVAILABLE:'), result.message
    before = len(h.node.cmd_vel_pub.messages)
    h.advance(1.)
    assert len(h.node.cmd_vel_pub.messages) == before


def test_new_action_requires_current_response_without_idle_dwell(runtime):
    h = runtime
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('TEST_STOP:')
    assert 'READY_CONFIRMING' not in states(h)
    assert any(x or z for t, state, x, z, phase in h.trace)


def test_two_actions_can_use_same_live_node_and_scan_health(runtime):
    h = runtime
    first = h.node.execute_callback(h.goal)
    assert first.message.startswith('TEST_STOP:')
    # Simulate the operator/mission creating a new action; no node restart.
    start = h.now
    h.goal = Goal()
    h.position = lambda t: (350., min(435., 280. + max(0., t - start - 1) * 45.))
    second = h.node.execute_callback(h.goal)
    assert second.message.startswith('TEST_STOP:'), second.message
    assert not h.node._owns_cmd_vel


@pytest.mark.parametrize('cost', [.000001, .0001, .001, .003])
def test_variable_intra_tick_processing_costs_do_not_manufacture_raw_stale(cost, runtime):
    h = runtime
    def clock():
        h.now += cost
        return h.now
    h.mod.time = NS(monotonic=clock)
    result = h.node.execute_callback(h.goal)
    assert result.message.startswith('TEST_STOP:'), result.message
    assert 'RAW_STALE' not in states(h)
    assert 'RAW_RECEIPT_CLOCK_ORDER' not in states(h)
    assert not h.node._owns_cmd_vel


def test_last_fault_is_preserved_in_terminal_result(runtime):
    h = runtime
    h.position = lambda t: (350., 300.)
    h.env_hook = lambda: setattr(h, 'scan_on', False) if h.now > 1.2 else None
    result = h.node.execute_callback(h.goal)
    assert 'SAFETY_UNAVAILABLE:' in result.message
    assert 'cause=MONITOR_INPUT_UNAVAILABLE' in result.message


def test_final_approach_with_progressing_clock_and_safe_inputs(runtime):
    h = runtime
    h.node.config = TrackingConfig(stop_only_test_mode=False, final_approach_calibrated=True)
    def clock():
        h.now += .0001
        return h.now
    h.mod.time = NS(monotonic=clock)
    result = h.node.execute_callback(h.goal)
    assert result.success and h.goal.state == 'SUCCEEDED', result.message
    assert 'RAW_STALE' not in states(h)
    final_times = [row[0] for row in h.trace if row[2] == .03 and row[4] == Phase.FINAL_APPROACH]
    assert final_times and h.now - final_times[0] >= 10.
