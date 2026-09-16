"""Regression tests for actual production pure classes; no ROS/hardware needed."""
from dataclasses import replace, fields
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.collision_safety import (  # noqa: E402
    CollisionConfig, CollisionSafety, compatible,
)
from navigation.tracking_control import (  # noqa: E402
    Command, Observation, Phase, TrackingConfig, TrackingController, VisionHealth,
)


ZERO = Command()
RAW = Command(0.02, 0.0)


def sample(s, t, raw=RAW, safe=None, phase='APPROACH', env=True, reason=''):
    request_id = s.request(raw, phase, t)
    if safe is not None:
        s.accept_safe(safe, t + 0.001, request_id, env)
    return s.evaluate(t + 0.002)


def observation(seq, t, bottom=300.0, x=350.0, cls=0):
    return Observation(seq, t, cls, .9, x, bottom - 25., 40., 50.)


@pytest.mark.parametrize('name,value', [
    ('safe_timeout_sec', 0), ('raw_timeout_sec', -1), ('safe_timeout_sec', math.nan),
    ('hold_timeout_sec', math.inf), ('hold_clear_frames', 0), ('hold_clear_frames', 2.2),
    ('hold_clear_ratio', 1.2), ('clear_min_interval_sec', 0), ('ready_timeout_sec', 0),
    ('watchdog_period_sec', .5), ('linear_zero_threshold', True),
])
def test_config_rejects_invalid(name, value):
    with pytest.raises(ValueError):
        replace(CollisionConfig(), **{name: value})


@pytest.mark.parametrize('raw,safe,expected', [
    (RAW, RAW, True), (RAW, ZERO, True), (RAW, Command(.01, 0), True),
    (RAW, Command(.03, 0), False), (RAW, Command(-.01, 0), False),
    (RAW, Command(.01, .1), False), (Command(0, .1), Command(0, .05), True),
    (Command(.08, .04), Command(.04, .02), True),
    (Command(.08, .04), Command(.04, 0), False),
    (ZERO, Command(.01, 0), False), (RAW, Command(math.nan, 0), False),
])
def test_safe_must_be_uniform_reduction(raw, safe, expected):
    assert compatible(raw, safe, 1e-5) == expected


def test_no_inputs_do_not_move():
    s = CollisionSafety(CollisionConfig())
    d = s.evaluate(0)
    assert d.command == ZERO and d.pause


def test_normal_and_slowing():
    s = CollisionSafety(CollisionConfig())
    assert sample(s, 0, safe=RAW).state == 'CLEAR'
    d = sample(s, .1, safe=Command(.01, 0))
    assert d.state == 'COLLISION_SLOWING' and d.command.linear_x == .01


@pytest.mark.parametrize('linear,angular', [(.001, 0), (.0001, 0), (0, .004)])
def test_near_zero_stops(linear, angular):
    s = CollisionSafety(CollisionConfig())
    raw = RAW if linear else Command(0, .1)
    d = sample(s, 0, raw, Command(linear, angular), phase='ALIGN' if angular else 'APPROACH')
    assert d.command == ZERO and d.state == 'COLLISION_HOLD'


def test_moving_arc_keeps_curvature():
    s = CollisionSafety(CollisionConfig())
    # Dropping this angular component would turn a checked curve into straight motion.
    d = sample(s, 0, Command(.04, .006), Command(.02, .003))
    assert d.command == Command(.02, .003)


def test_identical_raw_callback_does_not_create_wait():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=RAW)
    s.request(RAW, 'APPROACH', .1)
    assert s.evaluate(.11).command == RAW


def test_changed_raw_and_old_incompatible_safe_wait():
    s = CollisionSafety(CollisionConfig())
    old = s.request(RAW, 'APPROACH', 0)
    turn = Command(0, .1)
    current = s.request(turn, 'ALIGN', .1)
    assert not s.accept_safe(RAW, .11, old)
    assert s.evaluate(.12).command == ZERO
    assert s.accept_safe(turn, .13, current)
    assert s.evaluate(.14).command == turn


def test_raw_zero_cannot_be_undone_by_late_safe():
    s = CollisionSafety(CollisionConfig())
    old = s.request(RAW, 'APPROACH', 0)
    s.request(ZERO, 'APPROACH', .1)
    assert not s.accept_safe(RAW, .11, old)
    assert s.evaluate(.12).command == ZERO


def test_malformed_safe_latches():
    s = CollisionSafety(CollisionConfig())
    request_id = s.request(RAW, 'APPROACH', 0)
    s.accept_safe(Command(math.inf, 0), .01, request_id)
    assert s.evaluate(.02).failure == 'SAFETY_INVALID_OUTPUT'


@pytest.mark.parametrize('reason', ['SCAN_STALE', 'SCAN_TF_UNAVAILABLE',
                                  'COLLISION_GEOMETRY_NOT_READY', 'COLLISION_TOPIC_OWNERSHIP'])
def test_environment_faults_stop_and_are_bounded(reason):
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=RAW)
    assert sample(s, .1, safe=RAW, env=False, reason=reason).command == ZERO
    for i in range(2, 35):
        d = sample(s, i / 10, safe=RAW, env=False, reason=reason)
    assert d.failure == 'SAFETY_UNAVAILABLE'


def test_watchdog_raw_stale():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=RAW)
    assert s.evaluate(.6).failure == 'SAFETY_UNAVAILABLE'
    assert s.last_fault_reason == 'RAW_STALE'
    assert s.evaluate(.6).command == ZERO


def test_watchdog_safe_stale():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=RAW)
    s.request(RAW, 'APPROACH', .4)
    assert s.evaluate(.61).failure == 'SAFETY_UNAVAILABLE'
    assert s.last_fault_reason == 'SAFE_STALE'


def test_boundary_jitter_blocked_and_latched():
    s = CollisionSafety(CollisionConfig())
    values = [0, .003, 0, .004, .001, .003]
    for i in range(33):
        d = sample(s, i / 10, safe=Command(values[i % len(values)], 0))
        assert d.command == ZERO
        if i < 30:
            assert d.state == 'COLLISION_HOLD'
    assert d.failure == 'COLLISION_BLOCKED'
    assert sample(s, 3.4, safe=RAW).failure == 'COLLISION_BLOCKED'


def test_hold_deadline_not_reset_by_zero_probe():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=ZERO)
    for i in range(1, 33):
        d = sample(s, i / 10, raw=ZERO, safe=ZERO)
    assert d.failure == 'COLLISION_BLOCKED'


def test_missing_safe_is_not_fresh_evidence_of_a_wall():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=ZERO)
    for i in range(1, 40):
        d = sample(s, i / 10, safe=None)
    assert d.failure == 'SAFETY_UNAVAILABLE'
    assert s.last_fault_reason == 'SAFE_STALE'


def test_hold_requires_three_new_spaced_clear_samples():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=ZERO)
    assert sample(s, .1, safe=Command(.016, 0)).state == 'COLLISION_HOLD'
    # Control ticks are not three independent clear results.
    for t in [.12, .14, .16]:
        assert s.evaluate(t).state == 'COLLISION_HOLD'
    assert sample(s, .2, safe=Command(.017, 0)).state == 'COLLISION_HOLD'
    d = sample(s, .3, safe=RAW)
    assert d.realign and d.command == ZERO
    assert sample(s, .4, safe=RAW).realign


def test_clear_burst_not_counted():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=ZERO)
    for t in [.1, .101, .102]:
        d = sample(s, t, safe=RAW)
    assert not d.realign


def test_clear_streak_resets_below_threshold():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=ZERO)
    sample(s, .1, safe=RAW)
    sample(s, .2, safe=Command(.015, 0))
    sample(s, .3, safe=RAW)
    assert not sample(s, .4, safe=RAW).realign
    assert sample(s, .5, safe=RAW).realign


def test_realign_ack_discards_old_safe():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, safe=ZERO)
    for t in [.1, .2, .3]:
        sample(s, t, safe=RAW)
    old = s.request(RAW, 'APPROACH', .31)
    assert s.acknowledge_realign(.35)
    current = s.request(Command(0, .05), 'REALIGN', .4)
    assert s.evaluate(.41).command == ZERO
    assert not s.accept_safe(RAW, .42, old)
    assert s.accept_safe(Command(0, .05), .43, current)
    assert s.evaluate(.44).command == Command(0, .05)


@pytest.mark.parametrize('safe', [Command(.029, 0), ZERO, Command(.001, 0)])
def test_final_any_meaningful_modification_latches(safe):
    s = CollisionSafety(CollisionConfig())
    d = sample(s, 0, Command(.03, 0), safe, 'FINAL_APPROACH')
    assert d.failure == 'FINAL_APPROACH_INTERRUPTED'
    assert sample(s, .1, Command(.03, 0), Command(.03, 0), 'FINAL_APPROACH').command == ZERO


def test_final_transport_handshake_does_not_start_clock():
    s = CollisionSafety(CollisionConfig())
    request_id = s.request(Command(.03, 0), 'FINAL_APPROACH', 0)
    assert not s.evaluate(.1).failure
    s.accept_safe(Command(.03, 0), .2, request_id)
    assert s.evaluate(.21).command.linear_x == .03
    s.mark_final_started()
    s.request(Command(.03, 0), 'FINAL_APPROACH', .8)
    assert s.evaluate(.81).failure == 'FINAL_APPROACH_INTERRUPTED'


def test_final_scan_interruption_before_completion():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0, Command(.03, 0), Command(.03, 0), 'FINAL_APPROACH')
    s.mark_final_started()
    d = sample(s, .1, Command(.03, 0), Command(.03, 0), 'FINAL_APPROACH', False, 'SCAN_STALE')
    assert d.failure == 'FINAL_APPROACH_INTERRUPTED'


def test_monitor_fault_retires_attempt_even_if_later_zero_responses_are_healthy():
    s = CollisionSafety(CollisionConfig())
    assert sample(s, 0, ZERO, ZERO, env=False).failure == 'SAFETY_UNAVAILABLE'
    for t in [.1, .2, .3, .4, .5, .7]:
        d = sample(s, t, ZERO, ZERO)
        assert d.failure == 'SAFETY_UNAVAILABLE' and d.command == ZERO


def make_approaching(collection=False, deferred=False):
    cfg = replace(TrackingConfig(), stop_only_test_mode=not collection,
                  final_approach_calibrated=collection)
    c = TrackingController(cfg, 0, 0, defer_final_motion=deferred)
    for i in range(3):
        c.observe(observation(i, .2 * i))
    assert c.phase == Phase.APPROACH
    return c


def test_pause_prevents_unsafe_final_and_requires_fresh_realign():
    c = make_approaching(collection=True, deferred=True)
    original_start = c.approach_started_at
    c.pause_for_safety()
    for i in range(3, 8):
        c.observe(observation(i, .2 * i, bottom=435))
        assert c.step(.2 * i) == ZERO
    assert c.phase == Phase.APPROACH
    assert c.resume_after_safety(1.5)
    assert c.step(1.51) == ZERO
    assert c.phase == Phase.REALIGN
    assert c.approach_started_at == original_start
    c.observe(observation(8, 1.6, bottom=300, x=300))
    assert c.step(1.61).angular_z > 0


def test_pause_does_not_claim_test_stop():
    c = make_approaching()
    c.pause_for_safety()
    c.observe(observation(4, .6, bottom=435))
    c.observe(observation(5, .8, bottom=435))
    assert not c.done


def test_final_duration_starts_with_approved_motion():
    c = make_approaching(collection=True, deferred=True)
    for seq, t in [(3, .6), (4, .8)]:
        c.observe(observation(seq, t, bottom=435))
    assert c.phase == Phase.FINAL_APPROACH
    # Delayed safety approval must not count as movement.
    for i in range(5, 62):
        t = i * .2
        c.observe(observation(i, t, cls=-1))
        c.step(t)
    assert c.phase == Phase.FINAL_APPROACH
    c.arm_final_motion(12.4)
    for i in range(62, 112):
        c.observe(observation(i, i * .2, cls=-1))
        c.step(i * .2)
    assert c.phase == Phase.FINAL_APPROACH
    c.observe(observation(112, 22.4, cls=-1))
    assert c.step(22.41) == ZERO
    assert c.phase == Phase.SUCCEEDED


def test_basic_alignment_near_stop_and_lost_regression():
    c = make_approaching()
    c.observe(observation(3, .6, bottom=432))
    assert c.step(.61) == ZERO
    c.observe(observation(4, .8, cls=-1))
    assert c.phase == Phase.TEST_COMPLETE
    c = make_approaching()
    for i in range(3, 15):
        c.observe(observation(i, i * .2, cls=-1))
    assert c.reason.startswith('LOST_TARGET:')


def test_slow_vision_progress_regression():
    c = TrackingController(TrackingConfig(), 0, 0)
    for i in range(3):
        c.observe(observation(i, i * .9, bottom=280))
    for i in range(3, 42):
        c.observe(observation(i, i * .9, bottom=280 + 4 * (i - 2)))
        c.step(i * .9)
    assert c.phase == Phase.TEST_COMPLETE


def test_shipped_config_and_launch():
    root = Path(__file__).resolve().parents[1]
    doc = yaml.safe_load((root / 'config/recycle_tracking.yaml').read_text())
    node_cfg = doc['recycle_tracking_node']['ros__parameters']
    t = TrackingConfig(**{f.name: node_cfg[f.name] for f in fields(TrackingConfig())})
    c = CollisionConfig(**{f.name: node_cfg['collision_' + f.name] for f in fields(CollisionConfig())})
    assert t.align_reference_x == 350 and t.lost_abort_frames == 12
    assert t.approach_far_speed == .08 and not t.stop_only_test_mode
    assert t.final_approach_calibrated and t.final_approach_duration_sec == 10
    assert c.hold_timeout_sec == 3
    cm = doc['tracking_collision_monitor']['ros__parameters']
    assert cm['cmd_vel_out_topic'] != '/cmd_vel'
    assert cm['HardStop']['max_points'] == 0
    assert cm['FootprintApproach']['max_points'] == 3
    assert cm['HardStop']['points'] == [.335, .15, .335, -.15, -.206, -.15, -.206, .15]
    assert cm['FootprintApproach']['points'] == [.325, -.14, .325, .14, -.196, .14, -.196, -.14]
    assert cm['source_timeout'] == .4 and cm['scan']['topic'] == '/scan'
    assert cm['base_shift_correction'] is False
    launch = (root / 'launch/navigation.launch.py').read_text()
    assert "executable='tracking_collision_monitor'" in launch
    assert 'nav2_lifecycle_manager' in launch
    assert 'return [collision, lifecycle, tracking]' in launch
    assert 'navigation.launch.py' in (root / 'launch/main.launch.py').read_text()
    deps = {el.text for el in ET.parse(root / 'package.xml').getroot() if el.tag in ('depend', 'exec_depend')}
    assert {'sensor_msgs', 'nav2_collision_monitor', 'navigation_interface', 'nav2_lifecycle_manager'} <= deps
