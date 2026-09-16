"""Input behavior at its actual owner (ROS Monitor), plus the request-ID gate.

Former ScanInput/InputRecovery tests lived in the Tracking process. The scan
ordering, last-good and clock cases now execute the C++ consumer over DDS;
health dwell/restart-counter assertions are replaced by observable stopping.
"""
import math
import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.collision_safety import CollisionConfig, CollisionSafety
from navigation.tracking_control import Command
from test_collision_models_ros import monitors  # noqa: F401

ROS = pytest.mark.skipif(os.environ.get('RECYCLE_ROS_PROBE') != '1',
                         reason='opt-in isolated ROS2 input probe')
RAW, ZERO = Command(.08), Command()
WALL = [(0.32, -.4, .32, .4)]


@ROS
@pytest.mark.parametrize('fault', ['old', 'future', 'zero_stamp', 'negative_stamp',
    'bad_nanosec', 'duplicate', 'out_of_order', 'empty', 'infinite', 'nan',
    'range_bounds', 'angle_nan', 'angle_step', 'empty_frame', 'new_frame'])
def test_bad_scan_does_not_clear_wall_or_renew_last_good_input(monitors, fault):
    with monitors() as p:
        assert p.sample((.08, 0.), WALL) == (0., 0.)
        kwargs = {}
        if fault == 'old': kwargs['stamp_offset'] = -2.
        elif fault == 'future': kwargs['stamp_offset'] = 2.
        elif fault == 'zero_stamp': kwargs['fixed_stamp'] = 0
        elif fault == 'negative_stamp': kwargs['fixed_stamp'] = -1
        elif fault == 'bad_nanosec': kwargs['scan_mutator'] = lambda s: setattr(s.header.stamp, 'nanosec', 1000000000)
        elif fault == 'duplicate': kwargs['fixed_stamp'] = p.stamp
        elif fault == 'out_of_order': kwargs['fixed_stamp'] = p.stamp - 50000000
        elif fault == 'empty': kwargs['malformed'] = True
        elif fault == 'infinite': kwargs['scan_mutator'] = lambda s: setattr(s, 'ranges', [math.inf] * 10)
        elif fault == 'nan': kwargs['scan_mutator'] = lambda s: setattr(s, 'ranges', [math.nan] * 10)
        elif fault == 'range_bounds': kwargs['scan_mutator'] = lambda s: setattr(s, 'range_max', 0.)
        elif fault == 'angle_nan': kwargs['scan_mutator'] = lambda s: setattr(s, 'angle_min', math.nan)
        elif fault == 'angle_step': kwargs['scan_mutator'] = lambda s: setattr(s, 'angle_increment', 0.)
        elif fault == 'empty_frame': kwargs['frame'] = ''
        elif fault == 'new_frame': kwargs['frame'] = 'missing_new_frame'
        # Rejected data cannot immediately replace a valid obstacle sample.
        assert p.sample((.08, 0.), duration=.12, **kwargs) == (0., 0.)
        assert p.last.input_valid
        # Continuous bad arrivals cannot turn into permission to move.
        assert p.sample((.08, 0.), duration=.65, **kwargs) == (0., 0.)
        assert not p.last.input_valid
        # Recovery is one fresh usable scan, without a restart/dwell FSM.
        assert p.sample((.08, 0.), duration=.65) == (.08, 0.)
        assert p.last.input_valid


@ROS
def test_monitor_retains_valid_scan_across_interleaved_replays(monitors):
    with monitors() as p:
        for _ in range(6):
            assert p.sample((.08, 0.), duration=.12) == (.08, 0.)
            assert p.sample((.08, 0.), fixed_stamp=p.stamp, duration=.10) == (.08, 0.)
            assert p.last.input_valid


def test_new_attempt_cannot_consume_previous_attempts_identical_safe_reply():
    old, new = CollisionSafety(CollisionConfig()), CollisionSafety(CollisionConfig())
    old_id = old.request(RAW, 'APPROACH', 0.)
    new_id = new.request(RAW, 'APPROACH', .1)
    assert old_id != new_id
    assert not new.accept_safe(RAW, .11, old_id)
    assert new.evaluate(.12).command == ZERO
    assert new.accept_safe(RAW, .13, new_id)
    assert new.evaluate(.14).command == RAW


@pytest.mark.parametrize('reply', ['duplicate', 'unknown', 'out_of_order', 'expired'])
def test_unmatched_response_cannot_overwrite_stop_or_extend_validity(reply):
    s = CollisionSafety(CollisionConfig())
    first = s.request(RAW, 'APPROACH', 0.)
    second = s.request(RAW, 'APPROACH', .1)
    assert s.accept_safe(ZERO, .11, second)
    request_id = {'duplicate': second, 'unknown': second + 100,
                  'out_of_order': first, 'expired': first}[reply]
    at = .7 if reply == 'expired' else .12
    assert not s.accept_safe(RAW, at, request_id)
    assert s.evaluate(at + .01).command == ZERO


@pytest.mark.parametrize('raw', [RAW, ZERO])
def test_monitor_input_fault_is_terminal_and_keeps_primary_cause(raw):
    s = CollisionSafety(CollisionConfig())
    request_id = s.request(raw, 'APPROACH', 0.)
    s.accept_safe(ZERO, .01, request_id, input_valid=False)
    assert s.evaluate(.02).failure == 'SAFETY_UNAVAILABLE'
    for i in range(1, 10):
        request_id = s.request(raw, 'APPROACH', i / 10)
        s.accept_safe(raw, i / 10 + .001, request_id)
        assert s.evaluate(i / 10 + .002).command == ZERO
    assert s.last_fault_reason == 'MONITOR_INPUT_UNAVAILABLE'


def test_loss_of_responses_outranks_obstacle_hold_deadline():
    s = CollisionSafety(CollisionConfig())
    request_id = s.request(RAW, 'APPROACH', 0.)
    s.accept_safe(ZERO, .01, request_id)
    assert s.evaluate(.02).state == 'COLLISION_HOLD'
    for i in range(1, 40):
        s.request(RAW, 'APPROACH', i / 10)
        decision = s.evaluate(i / 10 + .01)
    assert decision.failure == 'SAFETY_UNAVAILABLE'
    assert s.last_fault_reason == 'SAFE_STALE'


def test_healthy_processing_jitter_does_not_create_negative_raw_age():
    import random
    rng = random.Random(20260916)
    s = CollisionSafety(CollisionConfig())
    for i in range(600):
        now = i / 10 + rng.uniform(0., .01)
        request_id = s.request(RAW, 'APPROACH', now)
        s.accept_safe(RAW, now + .002, request_id)
        assert s.evaluate(now + .003).command == RAW


def test_clock_reversal_never_authorizes_motion():
    s = CollisionSafety(CollisionConfig())
    request_id = s.request(RAW, 'APPROACH', 10.)
    assert not s.accept_safe(RAW, 9., request_id)
    assert s.evaluate(9.).failure == 'SAFETY_UNAVAILABLE'


@ROS
def test_paused_ros_clock_does_not_keep_scan_alive(monitors):
    with monitors(sim_time=True) as p:
        assert p.sample((.08, 0.)) == (.08, 0.)
        p.clock_paused = True
        assert p.sample((.08, 0.), duration=.8) == (0., 0.)
        assert not p.last.input_valid
        p.clock_paused = False
        assert p.sample((.08, 0.), duration=.8) == (.08, 0.)


@ROS
def test_joint_clock_reset_can_accept_fresh_new_epoch_without_retimestamping(monitors):
    with monitors(sim_time=True) as p:
        assert p.sample((.08, 0.)) == (.08, 0.)
        old_stamp = p.stamp
        p.sim_clock -= 5.
        assert p.sample((.08, 0.), duration=.8) == (.08, 0.)
        assert p.last.input_valid and p.stamp < old_stamp


@ROS
def test_monitor_geometry_is_immutable_while_running(monitors):
    import rclpy
    from rclpy.parameter import Parameter
    from rcl_interfaces.srv import SetParameters
    with monitors() as p:
        client = p.node.create_client(SetParameters, '/tracking_collision_monitor/set_parameters')
        assert client.wait_for_service(timeout_sec=2.)
        for name, value in [('HardStop.enabled', False), ('scan.enabled', False),
                            ('source_timeout', 20.), ('FootprintApproach.points', [0., 0., 1., 0., 1., 1.])]:
            request = SetParameters.Request(parameters=[Parameter(name, value=value).to_parameter_msg()])
            future = client.call_async(request)
            rclpy.spin_until_future_complete(p.node, future, timeout_sec=2.)
            assert future.done() and not future.result().results[0].successful
        assert p.sample((.08, 0.), WALL) == (0., 0.)


@ROS
def test_monitor_lifecycle_stops_output_and_reactivation_needs_current_scan(monitors):
    import time
    import rclpy
    from lifecycle_msgs.msg import Transition
    from navigation_interface.msg import CollisionCommand
    from geometry_msgs.msg import Twist
    with monitors() as p:
        assert p.sample((.08, 0.)) == (.08, 0.)
        p.change_state(Transition.TRANSITION_DEACTIVATE)
        deadline = time.monotonic() + .6
        while time.monotonic() < deadline:
            rclpy.spin_once(p.node, timeout_sec=.05)  # Drain earlier replies; let scan expire.
        before = len(p.replies)
        velocity = Twist(); velocity.linear.x = .08
        for i in range(5):
            p.raw_pub.publish(CollisionCommand(request_id=10000+i, velocity=velocity))
            rclpy.spin_once(p.node, timeout_sec=.05)
        assert len(p.replies) == before
        p.change_state(Transition.TRANSITION_ACTIVATE)
        assert p.sample((.08, 0.), scan_on=False) == (0., 0.)
        assert not p.last.input_valid
        assert p.sample((.08, 0.)) == (.08, 0.)
