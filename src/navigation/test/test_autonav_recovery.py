"""Real AutoNav methods with ROS service/action doubles. NOT an on-robot test."""
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.patrol_recovery import AUTO_PATROL_REASONS  # noqa: E402
from test_tracking_collision_adapter import runtime as tracking_runtime  # noqa: F401,E402
from test_tracking_collision_adapter import Twist  # noqa: E402


@pytest.fixture
def nav_runtime(monkeypatch):
    h = NS(now=1., health_on=True, common=True, collision=True, released=True, cleanup=True,
           scan_on=True, scan_sources=1, scan_offset=0.,
           wheel_names=['auto_nav', 'controller_server', 'recycle_tracking_node', 'recycle'],
           odom_on=True, x=0., y=0., linear=0., angular=0., seq=0)

    class Publisher:
        def __init__(self, topic):
            self.topic, self.messages = topic, []
        def publish(self, msg):
            self.messages.append(msg)

    class Service:
        def __init__(self, name):
            self.name = name
            self.ready = True
            self.answer = True
            self.success = True
            self.state = 3
            self.calls = []
        def service_is_ready(self):
            return self.ready
        def call_async(self, req):
            f = Future()
            self.calls.append((req, f))
            if self.answer:
                f.set_result(NS(success=self.success, reason='OK',
                                current_state=NS(id=self.state)))
            return f

    class Action:
        def __init__(self, node, typ, name):
            self.ready, self.calls, self.name = True, [], name
            self.waits = 0
            node.actions[name] = self
        def wait_for_server(self, **kw):
            self.waits += 1
            return self.ready
        def server_is_ready(self):
            return self.ready
        def send_goal_async(self, goal, **kw):
            f = Future()
            self.calls.append((goal, f))
            return f

    class FakeNode:
        def __init__(self, name):
            self.name, self.params, self.pubs = name, {}, {}
            self.subs, self.services, self.actions, self.clients = {}, {}, {}, {}
            self.logs, self.timers = [], []
        def declare_parameter(self, name, value, *args):
            self.params.setdefault(name, value)
        def get_parameter(self, name):
            return NS(value=self.params[name])
        def create_publisher(self, typ, topic, qos):
            p = Publisher(topic)
            self.pubs[topic] = p
            return p
        def create_subscription(self, typ, topic, cb, qos):
            self.subs[topic] = cb
            return NS()
        def create_client(self, typ, name):
            c = Service(name)
            self.clients[name] = c
            return c
        def create_service(self, typ, name, cb):
            self.services[name] = cb
            return NS()
        def create_timer(self, period, cb, **kw):
            t = NS(cancel=lambda: None, callback=cb)
            self.timers.append(t)
            return t
        def destroy_timer(self, timer):
            pass
        def get_logger(self):
            return NS(info=self.logs.append, warn=self.logs.append, error=self.logs.append)
        def get_clock(self):
            def now():
                ns = 1700000000 * 10**9 + round(h.now * 10**9)
                return NS(nanoseconds=ns, to_msg=lambda: NS(sec=ns // 10**9, nanosec=ns % 10**9))
            return NS(now=now)
        def get_publishers_info_by_topic(self, topic):
            names = ['lidar'] * h.scan_sources if topic == '/scan' else h.wheel_names
            return [NS(node_name=name) for name in names]

    class Buffer:
        def __init__(self):
            self.good = True
            self.scan_good = True
        def can_transform(self, *args):
            return self.good and (args[0] != 'base_footprint' or self.scan_good)

    def pose():
        return NS(header=NS(frame_id='', stamp=None), pose=NS(
            position=NS(x=0., y=0.), orientation=NS(w=1.)))

    modules = {
        'rclpy': dict(ok=lambda: True, shutdown=lambda: None),
        'rclpy.node': dict(Node=FakeNode),
        'rclpy.action': dict(ActionClient=Action),
        'rclpy.qos': dict(QoSProfile=lambda **kw: NS(**kw),
                           DurabilityPolicy=NS(TRANSIENT_LOCAL=1, VOLATILE=2),
                           ReliabilityPolicy=NS(RELIABLE=1, BEST_EFFORT=2)),
        'rclpy.clock': dict(Clock=lambda **kw: NS(**kw), ClockType=NS(STEADY_TIME=1)),
        'rclpy.time': dict(Time=lambda: NS()),
        'tf2_ros': dict(Buffer=Buffer, TransformListener=lambda *a: NS()),
        'nav2_msgs.action': dict(NavigateToPose=NS(Goal=lambda: NS())),
        'nav_msgs.msg': dict(Path=type('Path', (), {}), Odometry=type('Odometry', (), {})),
        'geometry_msgs.msg': dict(PoseStamped=pose, Twist=Twist),
        'sensor_msgs.msg': dict(LaserScan=type('LaserScan', (), {})),
        'action_msgs.msg': dict(GoalStatus=NS(STATUS_CANCELED=5, STATUS_SUCCEEDED=4, STATUS_ABORTED=6)),
        'std_msgs.msg': dict(String=lambda data='': NS(data=data)),
        'rcl_interfaces.msg': dict(ParameterDescriptor=lambda **kw: NS()),
        'std_srvs.srv': dict(Trigger=NS(Request=lambda: NS())),
        'lifecycle_msgs.srv': dict(GetState=NS(Request=lambda: NS())),
        'my_yolo_msgs.msg': dict(DetectedObject=type('DetectedObject', (), {})),
        'my_yolo_msgs.srv': dict(SetTracking=NS(Request=lambda: NS())),
        'navigation_interface.action': dict(RecycleActionMsg=NS(Goal=lambda: NS())),
        'navigation_interface.srv': dict(ControlServo=NS(Request=lambda: NS()),
                                          ControlPantilt=NS(Request=lambda: NS())),
    }
    for name, attrs in modules.items():
        m = ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
    path = Path(__file__).resolve().parents[1] / 'navigation/auto_nav.py'
    spec = importlib.util.spec_from_file_location('_autonav_recovery_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time = NS(monotonic=lambda: h.now)
    n = mod.AutoNav()
    h.node, h.mod = n, mod
    n.is_running = True
    n.resume_x, n.resume_y = 2., 0.
    n.waypoints = [(2., 0.), (3., 0.), (4., 0.)]
    n.current_idx = 0
    n.home_x, n.home_y = 4., 0.
    n.object_id = n.previous_object_id = 0
    n.target_x, n.target_y, n.target_h = 350., 300., 50.
    n.object_found = True
    n._servo_open = True

    def inputs():
        h.seq += 1
        if h.scan_on and h.common:
            ns = n.get_clock().now().nanoseconds + round(h.scan_offset * 1e9)
            n._patrol_scan_callback(NS(
                header=NS(stamp=NS(sec=ns // 10**9, nanosec=ns % 10**9), frame_id='base_scan'),
                angle_min=-1., angle_max=1., angle_increment=.1,
                range_min=.05, range_max=10., ranges=[1.] * 21))
        if h.health_on:
            health = dict(revision='patrol_recovery_v2', session='A', sequence=h.seq,
                          released=h.released, common_ready=h.common,
                          common_reason='' if h.common else 'SCAN_STALE',
                          collision_ready=h.collision, cleanup_ready=h.cleanup)
            n._health_callback(NS(data=json.dumps(health)))
        if h.odom_on:
            stamp = n.get_clock().now().to_msg()
            n._odom_callback(NS(header=NS(stamp=stamp, frame_id='odom'), child_frame_id='base_footprint',
                               pose=NS(pose=NS(position=NS(x=h.x, y=h.y))),
                               twist=NS(twist=NS(linear=NS(x=h.linear, y=0.),
                                                 angular=NS(z=h.angular)))))

    def advance(dt=.2, count=1):
        for _ in range(count):
            h.now = round(h.now + dt, 8)
            inputs()
            n._auto_return_tick()
    h.advance, h.inputs = advance, inputs
    h.goal_count = lambda: len(n._action_client.calls)
    h.states = lambda: [m.data for m in n._recovery_pub.messages]
    # Seed idle cleanup health, not stopped odometry evidence.
    h.odom_on = False
    h.inputs()
    h.odom_on = True
    return h


class Handle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancels = 0
    def get_result_async(self):
        return self.result
    def cancel_goal_async(self):
        self.cancels += 1
        f = Future()
        f.set_result(NS())
        return f


def detection(cls=0):
    return NS(id=cls, confidence=.9, coord=[350., 320., 60., 70.], min_y=200.)


@pytest.mark.parametrize('reason', sorted(AUTO_PATROL_REASONS))
def test_recoverable_failure_sends_one_patrol_goal_after_checks(nav_runtime, reason):
    h = nav_runtime
    n = h.node
    n._handle_tracking_failure(reason + ': test')
    assert n._return_plan is not None and n._tracking_safety_hold
    assert h.goal_count() == 0
    h.advance(count=12)
    assert h.goal_count() == 1
    assert not n._tracking_safety_hold
    assert n._acquisition_lock.locked
    assert n._nav_goal_pending
    assert n._action_client.calls[0][0].pose.pose.position.x == 2.
    assert n.collected_count == 0
    # No zero-command watchdog should fight a newly submitted Nav2 goal.
    before = len(n.cmd_vel_pub.messages)
    h.advance(count=10)
    assert h.goal_count() == 1 and len(n.cmd_vel_pub.messages) == before


def test_persistent_scan_failure_then_recovery_needs_no_manual_service(nav_runtime):
    h = nav_runtime
    h.common = False
    h.node._handle_tracking_failure('SAFETY_UNAVAILABLE: cause=SCAN_STALE')
    h.advance(count=60)
    assert h.goal_count() == 0 and h.node._return_plan is not None
    assert 'RECOVERY_WAIT_SCAN_STALE' in h.states()
    h.common = True
    h.advance(count=10)
    assert h.goal_count() == 1


def test_monitor_only_fault_allows_patrol_but_disables_new_collection(nav_runtime):
    h = nav_runtime
    h.collision = False
    h.node._handle_tracking_failure('SAFETY_UNAVAILABLE: cause=SAFE_STALE')
    h.advance(count=12)
    assert h.goal_count() == 1
    handle = Handle()
    h.node._action_client.calls[-1][1].set_result(handle)
    h.node._acquisition_lock.locked = False  # isolate monitor readiness gate
    h.advance(count=15)
    h.node.object_callback(detection())
    assert handle.cancels == 0
    h.collision = True
    h.advance(count=6)
    h.node.object_callback(detection())
    assert handle.cancels == 1


@pytest.mark.parametrize('reason', ['TEST_STOP', 'INTERNAL_ERROR',
                                   'SAFETY_INTERNAL_ERROR', 'SERVICE_ERROR', 'TRACKING_CANCELED'])
def test_terminal_inspection_cases_never_autorestart(nav_runtime, reason):
    h = nav_runtime
    h.node._handle_tracking_failure(reason + ': test')
    h.advance(count=40)
    assert h.goal_count() == 0
    assert h.node._return_plan is None
    assert h.node._tracking_safety_hold


def test_manual_reset_queues_same_health_gates_does_not_bypass_scan(nav_runtime):
    h = nav_runtime
    h.common = False
    h.node._handle_tracking_failure('TEST_STOP: test')
    response = h.node.reset_tracking_hold_callback(None, NS())
    assert response.success
    h.advance(count=20)
    assert h.goal_count() == 0
    h.common = True
    h.advance(count=10)
    assert h.goal_count() == 1


def test_missing_tracking_health_does_not_replace_scan_or_odom_checks(nav_runtime):
    h = nav_runtime
    h.health_on = False
    h.scan_on = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    assert h.goal_count() == 0
    h.scan_on = True
    h.odom_on = False
    h.advance(count=10)
    assert h.goal_count() == 0
    h.odom_on = True
    h.advance(count=10)
    assert h.goal_count() == 1


def test_map_tf_recovers_without_operator(nav_runtime):
    h = nav_runtime
    h.node._recovery_tf.good = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=20)
    assert h.goal_count() == 0
    h.node._recovery_tf.good = True
    h.advance(count=10)
    assert h.goal_count() == 1


def test_nav2_inactive_then_active_auto_resumes_without_changing_lifecycle(nav_runtime):
    h = nav_runtime
    n = h.node
    n._nav_state_clients['controller_server'].state = 2
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=20)
    assert h.goal_count() == 0
    n._nav_state_clients['controller_server'].state = 3
    h.advance(count=10)
    assert h.goal_count() == 1
    assert all(name.endswith('/get_state') for name in n.clients if 'controller' in name)


def test_missing_nav_state_service_keeps_polling_with_no_request_flood(nav_runtime):
    h = nav_runtime
    c = h.node._nav_state_clients['planner_server']
    c.answer = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=40)
    assert h.goal_count() == 0 and len(c.calls) <= 5
    c.answer = True
    h.advance(count=25)
    assert h.goal_count() == 1


def test_servo_response_required_and_service_recovery_is_automatic(nav_runtime):
    h = nav_runtime
    c = h.node.servo_client
    c.success = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=30)
    assert h.goal_count() == 0
    assert 2 <= len(c.calls) < 10
    c.success = True
    h.advance(count=30)
    assert h.goal_count() == 1 and not h.node._servo_open


def test_delayed_open_callback_does_not_retry_after_recovery_close(nav_runtime):
    h = nav_runtime
    n = h.node
    n.servo_client.answer = False
    n.trigger_servo_movement(-90, 90, verify=True)
    old_open = n.servo_future
    n.servo_client.answer = True
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    calls = len(n.servo_client.calls)
    old_open.set_result(NS(success=False, message='late failure'))
    assert len(n.servo_client.calls) == calls
    assert not n._servo_open


def test_stop_during_recovery_replaces_patrol_with_prepared_home(nav_runtime):
    h = nav_runtime
    h.common = False
    h.node._handle_tracking_failure('SAFETY_UNAVAILABLE: test')
    h.advance(count=2)
    h.node.command_callback(NS(data='STOP'))
    h.common = True
    h.advance(count=30)
    assert h.goal_count() == 1 and h.node._return_plan is None
    assert h.node._action_client.calls[0][0].pose.pose.position.x == h.node.home_x


def test_stop_after_goal_sent_cancels_late_accepted_goal(nav_runtime):
    h = nav_runtime
    n = h.node
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    assert n._nav_goal_pending
    n.command_callback(NS(data='STOP'))
    handle = Handle()
    n._action_client.calls[-1][1].set_result(handle)
    assert handle.cancels == 1
    h.advance(count=10)
    assert h.goal_count() == 1


def test_pending_action_or_tracking_ownership_blocks_handoff(nav_runtime):
    h = nav_runtime
    h.released = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    assert h.goal_count() == 0
    h.released = True
    h.node._tracking_goal_pending = True
    h.advance(count=10)
    assert h.goal_count() == 0
    h.node._tracking_goal_pending = False
    h.advance(count=10)
    assert h.goal_count() == 1


def test_actual_stopping_required_not_just_zero_intent(nav_runtime):
    h = nav_runtime
    h.linear = .05
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    assert h.goal_count() == 0
    h.linear = 0.
    h.advance(count=10)
    assert h.goal_count() == 1


def test_same_target_not_reacquired_until_waypoint_and_real_movement(nav_runtime):
    h = nav_runtime
    n = h.node
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=12)
    handle = Handle()
    n._action_client.calls[-1][1].set_result(handle)
    h.advance(count=20)  # time alone must not unlock
    n.object_callback(detection())
    assert handle.cancels == 0
    # Immediate SUCCEEDED at same spot must not unlock.
    handle.result.set_result(NS(status=4, result=NS()))
    assert n._acquisition_lock.locked
    next_handle = Handle()
    n._action_client.calls[-1][1].set_result(next_handle)
    for i in range(1, 13):
        h.x = i * .01
        h.advance(.1)
    n.object_callback(detection())
    assert next_handle.cancels == 0
    next_handle.result.set_result(NS(status=4, result=NS()))
    assert not n._acquisition_lock.locked
    assert 'COLLECTION_REARMED' in h.states()


def test_nav_abort_retries_do_not_unlock_collection(nav_runtime):
    h = nav_runtime
    n = h.node
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    handle = Handle()
    n._action_client.calls[-1][1].set_result(handle)
    handle.result.set_result(NS(status=6, result=NS()))
    assert n.abort_retry_count == 1 and n._acquisition_lock.locked
    assert h.goal_count() == 2  # team's existing Nav2 retry preserved


def test_full_basket_previous_class_not_erased_by_failure(nav_runtime):
    h = nav_runtime
    n = h.node
    n.collected_count, n.previous_object_id, n.object_id = 2, 0, 0
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=10)
    assert n.collected_count == 2 and n.previous_object_id == 0 and n.object_id == 0


def test_class_and_goal_snapshot_not_overwritten_while_tracking(nav_runtime):
    h = nav_runtime
    n = h.node
    n.object_found = True
    original = n.object_id, n.target_x, n.target_y, n.target_h
    n.object_callback(detection(2))
    assert (n.object_id, n.target_x, n.target_y, n.target_h) == original
    assert n.y_min == 200.  # basket observation deliberately still updates


def test_cleanup_unconfirmed_does_not_freeze_patrol_but_disables_collection(nav_runtime):
    h = nav_runtime
    n = h.node
    h.cleanup = False
    n._handle_tracking_failure('COLLISION_BLOCKED: test; CLEANUP_UNCONFIRMED: test')
    h.advance(count=15)
    assert h.goal_count() == 1 and not n._tracking_mode_clear
    h.cleanup = True
    h.advance(count=10)
    assert n._tracking_mode_clear
    assert 'set_tracking_mode' not in n.clients  # one ON/OFF owner in Tracking


def test_no_saved_patrol_goal_never_fabricates_one(nav_runtime):
    h = nav_runtime
    h.node.resume_x = None
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=20)
    assert h.goal_count() == 0 and h.node._tracking_safety_hold


def test_timer_uses_steady_time_and_service_names(nav_runtime):
    n = nav_runtime.node
    assert n._recovery_clock.clock_type == 1
    assert '/controller_server/get_state' in n.clients
    assert '/auto_nav/reset_tracking_hold' in n.services
    assert '/tracking_collision/health' in n.subs


def test_idle_monitor_probe_is_private_not_a_motion_command(tracking_runtime):
    h = tracking_runtime
    n = h.node
    h.now = .1
    h.environment()
    n._idle_health_probe()
    assert n._raw_pub.messages[-1].linear.x == 0.
    assert not n.cmd_vel_pub.messages
    n._safe_callback(Twist())
    n._publish_health()
    assert not n.cmd_vel_pub.messages
    row = json.loads(n._health_pub.messages[-1].data)
    assert row['common_ready'] and row['released']


def test_monitor_failure_common_health_stays_available_for_nav2(tracking_runtime):
    h = tracking_runtime
    n = h.node
    h.now = .1
    h.environment()
    n._geometry_at = {'approach': None, 'stop': None}
    n._publish_health()
    row = json.loads(n._health_pub.messages[-1].data)
    assert row['common_ready']
    assert not row['collision_ready']


def test_bad_scan_or_foreign_wheel_writer_blocks_common_health(tracking_runtime):
    h = tracking_runtime
    n = h.node
    h.now = .1
    h.environment()
    n.wheel_names = ['recycle_tracking_node', 'auto_nav', 'controller_server',
                     'recycle', 'teleop_twist_keyboard']
    n._publish_health()
    assert not json.loads(n._health_pub.messages[-1].data)['common_ready']
    n.wheel_names.remove('teleop_twist_keyboard')
    h.now = 2.
    n._publish_health()
    assert not json.loads(n._health_pub.messages[-1].data)['common_ready']


def test_idle_health_dwell_and_late_moving_safe_never_drives(tracking_runtime):
    h = tracking_runtime
    n = h.node
    for i in range(1, 10):
        h.now = i * .1
        h.environment()
        n._idle_health_probe()
        n._safe_callback(Twist())
        n._publish_health()
    assert json.loads(n._health_pub.messages[-1].data)['collision_ready']
    before = n._idle_safe_sequence
    moving = Twist()
    moving.linear.x = .08
    n._safe_callback(moving)
    assert n._idle_safe_sequence == before
    assert not n.cmd_vel_pub.messages


def test_active_goal_does_not_get_idle_probe(tracking_runtime):
    h = tracking_runtime
    n = h.node
    n._owns_cmd_vel = True
    count = len(n._raw_pub.messages)
    n._idle_health_probe()
    assert len(n._raw_pub.messages) == count


def test_manual_sensor_retry_cancels_auto_return_and_old_acquisition_lock(nav_runtime):
    h = nav_runtime
    n = h.node
    h.common = False
    n._handle_tracking_failure('SENSOR_STALE: test')
    h.advance(count=10)
    h.common = True
    # Deliver live data without running the auto-return timer yet.
    for i in range(6):
        h.now += .2
        h.inputs()
        n.object_callback(detection())
    response = n.resume_sensor_hold_callback(None, NS())
    assert response.success
    assert n._return_plan is None
    assert not n._acquisition_lock.locked
    assert n._tracking_goal_pending
    h.advance(count=10)
    assert h.goal_count() == 0
    assert len(n._recycle_tracking_client.calls) == 1


def test_busy_nav_request_prevents_duplicate_submission(nav_runtime):
    n = nav_runtime.node
    n.object_found = False
    n.send_goal(2., 0.)
    n.send_goal(3., 0.)
    assert len(n._action_client.calls) == 1


def test_operator_battery_low_also_cancels_auto_queue(nav_runtime):
    h = nav_runtime
    h.common = False
    h.node._handle_tracking_failure('SAFETY_NOT_READY: test')
    h.node.command_callback(NS(data='BATTERY_LOW'))
    h.common = True
    h.advance(count=20)
    assert h.goal_count() == 1
    assert h.node._action_client.calls[-1][0].pose.pose.position.x == h.node.home_x


def test_disabled_automatic_policy_keeps_explicit_reset_available(nav_runtime):
    from dataclasses import replace
    h = nav_runtime
    n = h.node
    n._recovery_cfg = replace(n._recovery_cfg, enabled=False)
    n._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=15)
    assert h.goal_count() == 0
    assert n.reset_tracking_hold_callback(None, NS()).success
    h.advance(count=15)
    assert h.goal_count() == 1


def test_close_service_timeout_is_retried_not_permanently_latched(nav_runtime):
    h = nav_runtime
    c = h.node.servo_client
    c.answer = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=40)
    assert h.goal_count() == 0
    assert len(c.calls) < 6
    c.answer = True
    h.advance(count=40)
    assert h.goal_count() == 1


def test_nav2_unreachable_server_prevents_return_then_recovers(nav_runtime):
    h = nav_runtime
    h.node._action_client.ready = False
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(count=15)
    assert h.goal_count() == 0
    h.node._action_client.ready = True
    h.advance(count=10)
    assert h.goal_count() == 1


def test_input_recovery_limit_retires_target_not_infinite_same_action(tracking_runtime):
    from navigation.tracking_control import TrackingController, TrackingConfig, Command
    h = tracking_runtime
    n = h.node
    n._owns_cmd_vel = True
    n._controller = TrackingController(TrackingConfig(), 0, 0.)
    for i in range(1, 4):
        h.now = i * .2
        h.environment()
        n._controller.pause_for_safety()
        n._safety.request(Command(.05, 0), 'APPROACH', h.now)
        n._safety.accept_safe(Command(.05, 0), h.now)
        n._safety.realign_required = True
        n._safety.last_fault_reason = 'SCAN_STALE'
        n._apply_safety_locked()
    assert n._input_auto_recoveries == 3
    assert n._safety.failure == 'SAFETY_UNAVAILABLE'
    assert 'INPUT_RECOVERY_LIMIT' in n._controller.reason
    assert n.cmd_vel_pub.messages[-1].linear.x == 0.


@pytest.mark.parametrize('fault', ['monitor', 'scan'])
def test_actual_tracking_result_and_health_drive_actual_autonav_return(
        nav_runtime, tracking_runtime, fault):
    """Cross-node scenario, no invented healthy status or duplicated FSM."""
    a, t = nav_runtime, tracking_runtime
    n, tracking = a.node, t.node
    t.monitor_on = fault != 'monitor'
    t.scan_on = fault != 'scan'
    result = tracking.execute_callback(t.goal)
    assert not result.success
    assert result.message.startswith('SAFETY_NOT_READY:')
    assert not tracking._owns_cmd_vel
    a.now = max(a.now, t.now) + .1  # both real nodes share a monotonic time axis
    f = Future()
    f.set_result(NS(status=6, result=result))
    n.recycle_tracking_result_callback(f)
    a.health_on = False  # receive only real adapter health below
    a.scan_on = False  # both nodes observe the same original scan messages
    # Keep fault for 2 seconds. Monitor-only failure allows otherwise healthy
    # patrol; missing common scan does not. No manual resume/reset is called.
    for i in range(60):
        t.now = a.now
        if i >= 20:
            t.monitor_on = t.scan_on = True
        t.environment()
        if t.scan_on:
            n._patrol_scan_callback(tracking._scan_pub.messages[-1])
        tracking._idle_health_probe()
        if t.monitor_on:
            tracking._safe_callback(Twist())
        tracking._publish_health()
        n._health_callback(tracking._health_pub.messages[-1])
        a.advance(.1)
        if i == 19:
            assert a.goal_count() == (1 if fault == 'monitor' else 0)
    assert a.goal_count() == 1
    assert n._acquisition_lock.locked
    # Idle health probes, including after recovery, must not drive the wheels.
    assert all(m.linear.x == 0 and m.angular.z == 0 for m in tracking.cmd_vel_pub.messages)


def test_one_scan_cannot_be_counted_again_by_fast_odom(nav_runtime):
    h = nav_runtime
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.inputs()
    h.scan_on = False
    h.advance(.1, count=7)
    assert h.goal_count() == 0


def test_one_stopped_odom_cannot_be_counted_again_by_scan_reports(nav_runtime):
    h = nav_runtime
    h.node._handle_tracking_failure('COLLISION_BLOCKED: test')
    h.advance(.1, count=2)
    h.odom_on = False
    h.advance(.1, count=5)
    assert h.goal_count() == 0
