"""Receipt-v1 regressions: ordering, last-good input, restart and recovery.

Pure Python. Tests use controlled MONOTONIC and ROS times separately; they do
not claim to validate DDS timing, TF transport, physical stopping distance or
an installed Nav2 binary.
"""
from dataclasses import replace
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.collision_safety import (  # noqa: E402
    CollisionConfig, CollisionSafety, InputRecovery, ScanInput,
)
from navigation.tracking_control import Command  # noqa: E402

NS = 10 ** 9
BASE = 1700000000 * NS
ZERO = Command()
RAW = Command(.08, 0.)


def offer(scan, t, delta=0., frame='base_scan', ros_t=None):
    stamp = BASE + round((t + delta) * NS)
    current = BASE + round((t if ros_t is None else ros_t) * NS)
    return scan.offer(stamp, frame, t, current)


def health(scan, t, ros_t=None):
    return scan.health(t, BASE + round((t if ros_t is None else ros_t) * NS))


def sample(s, t, raw=RAW, safe=RAW, scan_seq=None, ok=True, reason=''):
    s.request(raw, 'APPROACH', t)
    s.accept_safe(safe, t + .001)
    return s.evaluate(t + .002, ok, reason, round(t * 1000) if scan_seq is None else scan_seq)


def test_last_good_duplicate_does_not_set_input_fault():
    s = ScanInput(CollisionConfig())
    assert offer(s, .1)
    assert not offer(s, .2, delta=-.1)
    assert s.last_rejection == 'SCAN_DUPLICATE'
    assert s.received_at == .1 and s.sequence == 1
    assert health(s, .2) == (True, '')


def test_out_of_order_does_not_poison_recent_accepted_scan():
    s = ScanInput(CollisionConfig())
    offer(s, .2)
    assert not offer(s, .3, delta=-.15)
    assert s.sequence == 1 and s.stamp_ns == BASE + round(.2 * NS)
    assert health(s, .3)[0]
    assert offer(s, .4) and s.sequence == 2


def test_duplicate_stream_never_renews_valid_receipt_deadline():
    s = ScanInput(CollisionConfig())
    offer(s, .1)
    for t in [.2, .3, .4, .5, .6, .7, 1.]:
        assert not offer(s, t, delta=.1 - t)
    assert s.sequence == 1 and s.received_at == .1
    assert health(s, 1.)[1] == 'SCAN_STALE'


def test_one_malformed_scan_retains_good_then_real_outage_expires():
    s = ScanInput(CollisionConfig())
    offer(s, .1)
    s.reject('SCAN_INVALID')
    assert health(s, .2)[0]
    assert not health(s, .501)[0]
    assert offer(s, .6) and health(s, .6)[0]


@pytest.mark.parametrize('offset,reason', [(-2., 'SCAN_STAMP_OLD'), (2., 'SCAN_STAMP_FUTURE')])
def test_persistent_skew_is_diagnosed_not_retimestamped(offset, reason):
    s = ScanInput(CollisionConfig())
    for i in range(1, 20):
        assert not offer(s, i / 10, delta=offset)
        assert health(s, i / 10) == (False, reason)
    assert s.sequence == 0
    assert s.last_offered_age_sec == pytest.approx(-offset)


def test_bad_stamp_then_corrected_stream_recovers_without_object_reset():
    s = ScanInput(CollisionConfig())
    assert not offer(s, .1, delta=2.)
    assert offer(s, .2)
    assert health(s, .2)[0]


def test_bad_future_packet_does_not_advance_high_watermark():
    s = ScanInput(CollisionConfig())
    offer(s, .1)
    assert not offer(s, .2, delta=60.)
    assert health(s, .2)[0]
    assert offer(s, .3)  # no waiting until the poisoned +60s stamp
    assert s.sequence == 2


def test_old_packets_arriving_continuously_are_not_fresh():
    s = ScanInput(CollisionConfig())
    offer(s, .1)
    for i in range(1, 20):
        assert not offer(s, 1 + i / 10, delta=-1.)
    assert health(s, 3.) == (False, 'SCAN_STALE')


def test_pause_sim_clock_does_not_keep_duplicate_scan_alive():
    s = ScanInput(CollisionConfig())
    offer(s, 0.)
    for t in [.1, .2, .3, .4, .5, .6]:
        s.offer(BASE, 'base_scan', t, BASE)
    assert s.sequence == 1
    assert s.health(.6, BASE)[1] == 'SCAN_STALE'


def test_ros_clock_jump_back_reconfirms_valid_new_epoch():
    s = ScanInput(CollisionConfig())
    assert s.offer(BASE + 100 * NS, 'base_scan', 1., BASE + 100 * NS)
    # Both ROS clocks now show ~90s. No new stamp is falsified to bypass age.
    assert not s.offer(BASE + 90 * NS, 'base_scan', 1.1, BASE + 90 * NS)
    assert not s.offer(BASE + round(90.1 * NS), 'base_scan', 1.2, BASE + round(90.1 * NS))
    assert s.offer(BASE + round(90.2 * NS), 'base_scan', 1.3, BASE + round(90.2 * NS))
    assert s.sequence == 2
    assert s.health(1.3, BASE + round(90.2 * NS))[0]


def test_repeated_restart_stamp_cannot_reconfirm():
    s = ScanInput(CollisionConfig())
    s.offer(BASE + 100 * NS, 'base_scan', 1., BASE + 100 * NS)
    for t in [1.1, 1.2, 1.3, 1.4]:
        assert not s.offer(BASE + 90 * NS, 'base_scan', t, BASE + 90 * NS)
    assert s.sequence == 1
    assert s.restart_count == 1


def test_out_of_order_old_samples_do_not_become_a_new_epoch():
    s = ScanInput(CollisionConfig())
    offer(s, 10.)
    for t in [10.5, 10.6, 10.7]:
        assert not offer(s, t, delta=-2.)
    assert s.sequence == 1


def test_restart_burst_cannot_reconfirm():
    s = ScanInput(CollisionConfig())
    s.offer(BASE + 100 * NS, 'base_scan', 1., BASE + 100 * NS)
    for i in range(3):
        assert not s.offer(BASE + 90 * NS + i * 1000000, 'base_scan',
                           1.1 + i * .001, BASE + 90 * NS + i * 1000000)
    assert s.sequence == 1 and s.restart_count == 1


def test_new_frame_needs_confirmation_not_silent_swap():
    s = ScanInput(CollisionConfig())
    offer(s, .1)
    assert not offer(s, .2, frame='new_scan')
    assert s.frame == 'base_scan'
    assert not offer(s, .6, frame='new_scan')
    assert not offer(s, .7, frame='new_scan')
    assert offer(s, .8, frame='new_scan')
    assert s.frame == 'new_scan'


@pytest.mark.parametrize('stamp,frame', [(-1, 'base_scan'), (0, 'base_scan'),
                                        (True, 'base_scan'), (BASE, ''), (BASE, None)])
def test_bad_metadata_never_accepted(stamp, frame):
    s = ScanInput(CollisionConfig())
    assert not s.offer(stamp, frame, .1, BASE)
    assert s.sequence == 0


def test_receipt_clock_reversal_is_not_mislabelled_as_scan_age():
    s = ScanInput(CollisionConfig())
    offer(s, .2)
    assert not offer(s, .1)
    assert s.last_rejection == 'SCAN_RECEIPT_CLOCK_ORDER'
    assert health(s, .1)[1] == 'SCAN_RECEIPT_CLOCK_ORDER'


def test_recovery_dwell_and_distinct_inputs():
    r = InputRecovery(CollisionConfig())
    assert not r.push(0, 1, 1)
    assert not r.push(.1, 2, 2)
    assert not r.push(.2, 3, 3)  # three results, but dwell not yet elapsed
    assert r.push(.51, 4, 4)


def test_same_scan_repeated_in_timer_does_not_complete_recovery():
    r = InputRecovery(CollisionConfig())
    for i in range(20):
        assert not r.push(i * .1, 1, i + 1)
    assert r.scan_count == 1


def test_same_safe_repeated_in_timer_does_not_complete_recovery():
    r = InputRecovery(CollisionConfig())
    for i in range(20):
        assert not r.push(i * .1, i + 1, 1)
    assert r.safe_count == 1


def test_burst_receipts_do_not_complete_recovery():
    r = InputRecovery(CollisionConfig())
    for i in range(10):
        assert not r.push(i * .001, i + 1, i + 1)
    assert r.scan_count == r.safe_count == 1
    assert not r.push(.6, 10, 10)


def test_one_glitch_resets_confirmation_but_not_absolute_fault_deadline():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0.)
    sample(s, .1, ok=False, reason='SCAN_TF_UNAVAILABLE')
    for t in [.2, .3, .4]:
        assert not sample(s, t).realign
    sample(s, .5, ok=False, reason='SCAN_TF_UNAVAILABLE')
    assert s.fault_started_at == pytest.approx(.102)
    for t in [.6, .7, .8, .9, 1.]:
        assert not sample(s, t).realign
    assert sample(s, 1.2).realign


def test_sensor_recovery_does_not_require_unmodified_speed():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0.)
    sample(s, .1, ok=False, reason='SCAN_STALE')
    for t in [.2, .3, .4, .5, .6]:
        d = sample(s, t, safe=Command(.04, 0))
        assert d.command == ZERO and not d.failure
    d = sample(s, .8, safe=Command(.04, 0))
    assert d.realign  # sensor healthy; reduced speed is not sensor failure


def test_recovered_sensor_with_actual_obstacle_enters_hold():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0.)
    sample(s, .1, ok=False, reason='SCAN_STALE')
    for t in [.2, .3, .4, .5, .6, .8]:
        d = sample(s, t, safe=ZERO)
    assert d.state == 'COLLISION_HOLD' and not d.failure
    assert not s.recovering
    for i in range(9, 41):
        d = sample(s, i / 10, safe=ZERO)
    assert d.failure == 'COLLISION_BLOCKED'


def test_real_sensor_fault_outranks_expired_wall_hold():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0., safe=ZERO)
    for i in range(1, 34):
        d = sample(s, i / 10, ok=False, reason='SCAN_STALE')
    assert d.failure == 'SAFETY_UNAVAILABLE'
    assert s.last_fault_reason == 'SCAN_STALE'


def test_healthy_10hz_after_raw_recording_never_has_negative_age():
    s = CollisionSafety(CollisionConfig())
    for i in range(100):
        start = i / 10
        s.request(RAW, 'APPROACH', start + .0001)
        s.accept_safe(RAW, start + .002)
        d = s.evaluate(start + .003, scan_sequence=i + 1)
        assert d.state == 'APPROACH' and d.command == RAW and not d.failure


def test_future_ordering_is_a_distinct_diagnostic_not_fake_stale():
    s = CollisionSafety(CollisionConfig())
    s.request(RAW, 'APPROACH', 10.0001)
    assert s.evaluate(10.0).state == 'RAW_RECEIPT_CLOCK_ORDER'


@pytest.mark.parametrize('field,value', [('recovery_frames', 1), ('scan_restart_frames', 2.5),
                                         ('recovery_stable_sec', 3.)])
def test_config_recovery_must_be_feasible(field, value):
    with pytest.raises(ValueError):
        replace(CollisionConfig(), **{field: value})


def test_hold_can_clear_on_new_evidence_after_transient_sensor_gap():
    s = CollisionSafety(CollisionConfig())
    sample(s, 0., safe=ZERO)
    for i in range(5, 26):
        d = sample(s, i / 10, safe=ZERO, ok=False, reason='SCAN_STALE')
        assert not d.failure
    for i in range(26, 33):
        d = sample(s, i / 10, safe=RAW)
    assert d.realign and not d.failure and d.command == ZERO


def test_seeded_jitter_and_duplicate_scans_keep_healthy_chain_running():
    import random
    rng = random.Random(20260914)
    scan = ScanInput(CollisionConfig())
    s = CollisionSafety(CollisionConfig())
    for i in range(600):
        t = 1.0 + i * .1 + rng.uniform(0., .01)
        assert offer(scan, t, delta=-.05)
        if i % 3 == 0:
            assert not offer(scan, t + .001, delta=-.051)  # repeated last stamp
        if i % 7 == 0:
            scan.reject('SCAN_INVALID', t + .002)
        ok, why = health(scan, t + .003)
        s.request(RAW, 'APPROACH', t + .004)
        s.accept_safe(RAW, t + .005)
        d = s.evaluate(t + .006, ok, why, scan.sequence)
        assert d.command == RAW and not d.failure and not d.pause
