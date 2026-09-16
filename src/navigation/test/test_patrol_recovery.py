"""Pure policy regressions; no ROS, motion hardware or TEB simulated here."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.patrol_recovery import (  # noqa: E402
    AUTO_PATROL_REASONS, AcquisitionLock, OdomEvidence,
    PatrolRecoveryConfig,
)

CFG = PatrolRecoveryConfig()
BASE = 1700000000 * 10**9


def test_only_documented_recoverable_reasons():
    assert 'COLLISION_BLOCKED' in AUTO_PATROL_REASONS
    assert 'SAFETY_UNAVAILABLE' in AUTO_PATROL_REASONS
    assert 'SENSOR_STALE' in AUTO_PATROL_REASONS
    assert 'FINAL_APPROACH_INTERRUPTED' in AUTO_PATROL_REASONS
    assert not AUTO_PATROL_REASONS.intersection({
        'TEST_STOP', 'SAFETY_INTERNAL_ERROR',
        'INTERNAL_ERROR', 'STOP', 'TRACKING_CANCELED', 'SERVICE_ERROR', 'UNKNOWN',
    })


@pytest.mark.parametrize('kw', [dict(enabled=1), dict(odom_timeout_sec=0), dict(service_timeout_sec=-1),
                              dict(period_sec=1.), dict(service_retry_sec=0),
                              dict(release_distance_m=-.1), dict(stopped_linear_speed=float('nan'))])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        replace(CFG, **kw)


def put_odom(o, t, *, x=0., y=0., linear=0., angular=0., offset=0., frame='odom'):
    stamp = BASE + round((t + offset) * 10**9)
    return o.receive(t, BASE + round(t * 10**9), stamp, frame, x, y, linear, angular)


def test_valid_stationary_odom():
    o = OdomEvidence(CFG)
    assert put_odom(o, 1.)
    assert o.stopped(1.1, BASE + 1100000000)


@pytest.mark.parametrize('kw', [dict(linear=.1), dict(angular=.1)])
def test_moving_robot_not_ready_for_handoff(kw):
    o = OdomEvidence(CFG)
    assert put_odom(o, 1., **kw)
    assert not o.stopped(1.1, BASE + 1100000000)


def test_odom_missing_replayed_wrong_frame_or_clock_skew():
    o = OdomEvidence(CFG)
    assert not o.fresh(0, BASE)
    assert not put_odom(o, 1., frame='map')
    assert not put_odom(o, 1., offset=-2.)
    assert put_odom(o, 1.)
    assert not put_odom(o, 1.1, offset=-.1)
    assert not o.fresh(2., BASE + 2 * 10**9)


def test_odom_epoch_reset_can_recover_after_gap():
    o = OdomEvidence(CFG)
    assert o.receive(1., BASE + 100 * 10**9, BASE + 100 * 10**9,
                     'odom', 0., 0., 0., 0.)
    assert o.receive(2., BASE + 90 * 10**9, BASE + 90 * 10**9,
                     'odom', 0., 0., 0., 0.)
    assert o.fresh(2., BASE + 90 * 10**9)


def test_suppression_requires_real_displacement_and_success():
    a = AcquisitionLock(.1)
    a.engage()
    a.begin_patrol(0., 0., 0.)
    assert not a.waypoint_succeeded()  # instant goal already within tolerance
    for i in range(1, 13):
        a.observe(i * .01, 0., i * .1)
    assert a.locked  # odometry alone isn't goal completion
    assert a.waypoint_succeeded()
    assert not a.locked


def test_hiding_target_or_waiting_does_not_unlock():
    a = AcquisitionLock(.1)
    a.engage()
    a.begin_patrol(0., 0., 0.)
    for i in range(1, 30):
        a.observe(.001 if i % 2 else 0., 0., i * .1)
    assert not a.waypoint_succeeded()
    assert a.locked


def test_odom_jump_and_gap_do_not_count_as_real_travel():
    a = AcquisitionLock(.1)
    a.engage()
    a.begin_patrol(0., 0., 0.)
    a.observe(10., 0., .1)
    assert not a.waypoint_succeeded()
    a.observe(20., 0., 2.)
    assert not a.waypoint_succeeded()


def test_returning_to_same_spot_does_not_release_lock():
    a = AcquisitionLock(.1)
    a.engage()
    a.begin_patrol(0., 0., 0.)
    for i, x in enumerate([.05, .10, .15, .10, .05, 0.], 1):
        a.observe(x, 0., i * .1)
    assert not a.waypoint_succeeded()
