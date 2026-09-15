"""Pure policy regressions; no ROS, motion hardware or TEB simulated here."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.patrol_recovery import (  # noqa: E402
    AUTO_PATROL_REASONS, AcquisitionLock, HealthFeed, OdomEvidence,
    PatrolRecoveryConfig, StableReadiness,
)

CFG = PatrolRecoveryConfig()
BASE = 1700000000 * 10**9


def row(seq=1, session='A', **kw):
    return dict(revision='patrol_recovery_v2', session=session, sequence=seq,
                released=True, common_ready=True, collision_ready=True, **kw)


def test_only_documented_recoverable_reasons():
    assert 'COLLISION_BLOCKED' in AUTO_PATROL_REASONS
    assert 'SAFETY_UNAVAILABLE' in AUTO_PATROL_REASONS
    assert 'SENSOR_STALE' in AUTO_PATROL_REASONS
    assert 'FINAL_APPROACH_INTERRUPTED' in AUTO_PATROL_REASONS
    assert not AUTO_PATROL_REASONS.intersection({
        'TEST_STOP', 'SAFETY_INTERNAL_ERROR',
        'INTERNAL_ERROR', 'STOP', 'TRACKING_CANCELED', 'SERVICE_ERROR', 'UNKNOWN',
    })


@pytest.mark.parametrize('kw', [dict(enabled=1), dict(samples=1), dict(samples=3.5),
                              dict(period_sec=1.), dict(health_timeout_sec=0),
                              dict(release_distance_m=-.1), dict(stable_sec=float('nan'))])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        replace(CFG, **kw)


def test_dwell_requires_distinct_packets_and_elapsed_time():
    r = StableReadiness(CFG)
    assert not r.push(0, 1)
    assert not r.push(.2, 1)
    assert not r.push(.4, 1)
    assert not r.push(.6, 1)
    assert not r.push(.6, 2)
    assert r.push(.7, 3)


def test_failed_readiness_restarts_dwell_not_entire_mission():
    r = StableReadiness(CFG)
    for t, seq in [(0, 1), (.2, 2), (.4, 3)]:
        assert not r.push(t, seq)
    assert not r.push(.6, 4, False)
    assert not r.push(.7, 5)
    assert not r.push(.9, 6)
    assert r.push(1.3, 7)


def test_readiness_gap_must_reconfirm():
    r = StableReadiness(CFG)
    r.push(0., 1)
    r.push(.3, 2)
    assert r.push(.6, 3)
    assert not r.push(2.0, 4)


def test_collection_health_needs_new_session_handshake():
    f = HealthFeed(CFG)
    for t, seq in [(0., 1), (.3, 2), (.6, 3)]:
        assert f.receive(row(seq), t)
    assert f.collection_ready(.6)
    assert f.receive(row(1, 'B'), .7)
    assert not f.collection_ready(.7)


def test_duplicate_health_never_renews_timeout():
    f = HealthFeed(CFG)
    assert f.receive(row(), .1)
    for t in [.2, .4, 1., 2.]:
        assert not f.receive(row(), t)
    assert not f.fresh(2.)


def test_ready_health_at_same_timestamp_is_not_enough_to_drive():
    f = HealthFeed(CFG)
    for seq in [1, 2, 3]:
        f.receive(row(seq), .1)
    assert not f.collection_ready(.1)
    assert not f.collection_ready(2.)


def test_collision_fault_does_not_invalidate_common_scan_for_patrol():
    f = HealthFeed(CFG)
    r = row()
    r['collision_ready'] = False
    f.receive(r, .1)
    assert f.common_ready(.2)
    assert not f.collection_ready(.2)


def test_active_tracking_cannot_claim_handoff_complete():
    f = HealthFeed(CFG)
    r = row()
    r['released'] = False
    f.receive(r, .1)
    assert not f.common_ready(.2)


@pytest.mark.parametrize('kw', [dict(sequence=True), dict(sequence=0), dict(session=''),
                              dict(common_ready='true'), dict(revision='receipt_v1')])
def test_malformed_health_not_authority(kw):
    f = HealthFeed(CFG)
    r = row()
    r.update(kw)
    assert not f.receive(r, .1)
    assert not f.common_ready(.1)


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
