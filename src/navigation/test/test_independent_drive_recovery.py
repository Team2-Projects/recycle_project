"""Actual AutoNav transitions; only ROS, hardware and clock boundaries are fake."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from test_autonav_recovery import nav_runtime, Handle, detection  # noqa: F401
from test_async_mission_handoff import accept_tracking, outcome


@pytest.mark.parametrize('destination', ['STOP', 'BATTERY_LOW', 'PATROL'])
@pytest.mark.parametrize('tracked', [False, True])
def test_tracking_disappearance_does_not_block_released_driving(nav_runtime, destination, tracked):
    h, n = nav_runtime, nav_runtime.node
    handle = accept_tracking(h) if tracked else None
    h.health_on = False
    h.wheel_names.remove('recycle_tracking_node')
    n._recycle_tracking_client.ready = False
    if destination != 'PATROL':
        n.command_callback(NS(data=destination))
    if handle is not None:
        handle.result.set_result(outcome(message='LOST_TARGET: gone; CLEANUP_UNCONFIRMED: timeout'))
    elif destination == 'PATROL':
        n._handle_tracking_failure('LOST_TARGET: gone')
    h.advance(count=20)
    assert h.goal_count() == 1
    assert n._nav_goal_purpose == ('PATROL' if destination == 'PATROL' else 'HOME')
    assert not n._tracking_health.fresh(h.now)
    patrol = Handle()
    n._action_client.calls[-1][1].set_result(patrol)
    before = len(n._recycle_tracking_client.calls)
    n.object_callback(detection())
    h.advance(count=20)
    assert len(n._recycle_tracking_client.calls) == before
    assert h.goal_count() == 1 and patrol.cancels == 0


@pytest.mark.parametrize('pending', [False, True])
def test_no_timeout_forgets_unfinished_tracking(nav_runtime, pending):
    h, n = nav_runtime, nav_runtime.node
    n.launch_recycle_tracking_action()
    acceptance = n._recycle_tracking_client.calls[-1][1]
    handle = Handle()
    if not pending:
        acceptance.set_result(handle)
    h.health_on = False
    n.command_callback(NS(data='STOP'))
    close_count = len(n.servo_client.calls)
    h.advance(count=400)
    assert h.goal_count() == 0 and len(n.servo_client.calls) == close_count
    assert n._last_recovery_status == 'RECOVERY_WAIT_ACTION_FINISH'
    if pending:
        acceptance.set_result(handle)
    assert handle.cancels == 1
    handle.result.set_result(outcome(5, 'STOP'))
    h.advance(count=20)
    assert h.goal_count() == 1 and n._nav_goal_purpose == 'HOME'


def test_transport_exception_is_not_wheel_release_evidence(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    handle = accept_tracking(h)
    n.command_callback(NS(data='STOP'))
    handle.result.set_exception(RuntimeError('transport lost, terminal status unknown'))
    h.health_on = False
    h.advance(count=200)
    assert not n._tracking_release_confirmed
    assert h.goal_count() == 0
    assert n._last_recovery_status == 'RECOVERY_WAIT_TRACKING_RELEASE'


@pytest.mark.parametrize('fault', ['scan', 'old_scan', 'future_scan', 'scan_tf',
                                  'map_tf', 'odom', 'motion', 'nav2', 'servo',
                                  'scan_sources', 'wheel_writer'])
def test_direct_common_inputs_remain_required_without_tracking(nav_runtime, fault):
    h, n = nav_runtime, nav_runtime.node
    h.health_on = False
    if fault == 'scan':
        h.scan_on = False
    elif fault in ('old_scan', 'future_scan'):
        h.scan_offset = -2. if fault == 'old_scan' else 2.
    elif fault == 'scan_tf':
        n._recovery_tf.scan_good = False
    elif fault == 'map_tf':
        n._recovery_tf.good = False
    elif fault == 'odom':
        h.odom_on = False
    elif fault == 'motion':
        h.linear = .05
    elif fault == 'nav2':
        n._nav_state_clients['planner_server'].state = 2
    elif fault == 'servo':
        n.servo_client.answer = False
    elif fault == 'scan_sources':
        h.scan_sources = 2
    else:
        h.wheel_names.append('teleop_twist_keyboard')
    n.command_callback(NS(data='BATTERY_LOW'))
    h.advance(count=200)
    assert h.goal_count() == 0 and n._return_plan.destination == 'HOME'
    if fault not in ('odom', 'motion', 'servo'):
        assert n._return_plan.servo_confirmed  # closing does not depend on scan/Nav2
    h.scan_on = h.odom_on = True
    h.scan_offset = h.linear = 0.
    h.scan_sources = 1
    h.wheel_names = ['auto_nav', 'controller_server', 'recycle']
    n._recovery_tf.good = n._recovery_tf.scan_good = True
    n._nav_state_clients['planner_server'].state = 3
    n.servo_client.answer = True
    h.advance(count=40)
    assert h.goal_count() == 1 and n._nav_goal_purpose == 'HOME'


def test_tracker_common_health_cannot_veto_healthy_local_navigation(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    h.health_on = False
    n._health_callback(NS(data=json.dumps(dict(
        revision='patrol_recovery_v2', session='B', sequence=1,
        released=True, common_ready=False, collision_ready=False, cleanup_ready=False,
        common_reason='SCAN_STALE'))))
    n.command_callback(NS(data='STOP'))
    h.advance(count=20)
    assert h.goal_count() == 1


def test_optional_tracking_server_does_not_block_startup_or_leave_pending_goal(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    assert n._recycle_tracking_client.waits == 0
    n._recycle_tracking_client.ready = False
    n.launch_recycle_tracking_action()
    assert not n._recycle_tracking_client.calls and not n._tracking_goal_pending
    assert n._tracking_release_confirmed and n._return_plan.destination == 'PATROL'
    h.advance(count=15)
    assert h.goal_count() == 1


def test_wait_heartbeat_has_elapsed_time_without_repeated_logs_or_forced_drive(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    h.scan_on = False
    n.command_callback(NS(data='STOP'))
    created = n._return_plan.created_at
    h.advance(count=200)
    rows = [json.loads(m.data) for m in n._recovery_details_pub.messages]
    waiting = [r for r in rows if r['state'] == 'RECOVERY_WAIT_SCAN_STALE']
    assert 30 <= len(waiting) <= 42
    assert waiting[-1]['elapsed_sec'] == pytest.approx(h.now - created, abs=1.1)
    assert waiting[-1]['state_elapsed_sec'] > waiting[0]['state_elapsed_sec']
    assert waiting[-1]['destination'] == 'HOME' and waiting[-1]['reason'] == 'STOP'
    assert waiting[-1]['waiting'] and not waiting[-1]['manual']
    assert sum('Patrol recovery: RECOVERY_WAIT_SCAN_STALE' in line for line in n.logs) == 1
    assert h.goal_count() == 0 and n._return_plan is not None
    h.scan_on = True
    h.advance(count=20)
    assert h.goal_count() == 1
    assert json.loads(n._recovery_details_pub.messages[-1].data)['state'] == 'HOME_STARTING'


def test_manual_request_is_reported_but_does_not_bypass_inputs(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    h.scan_on = False
    n._handle_tracking_failure('TEST_STOP: operator inspection')
    assert n.reset_tracking_hold_callback(None, NS()).success
    h.advance(count=20)
    details = json.loads(n._recovery_details_pub.messages[-1].data)
    assert details['manual'] and details['destination'] == 'PATROL'
    assert details['elapsed_sec'] > 0 and h.goal_count() == 0


def test_normal_waypoints_do_not_enter_return_preparation(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    h.scan_on = h.health_on = h.odom_on = False
    n.servo_client.answer = False
    close_count = len(n.servo_client.calls)
    n.object_found = False
    n.send_next_goal()
    handle = Handle()
    n._action_client.calls[-1][1].set_result(handle)
    handle.result.set_result(outcome(4))
    assert h.goal_count() == 2 and n.current_idx == 1
    assert len(n.servo_client.calls) == close_count and n._return_plan is None


def test_scan_policy_values_match_existing_tracking_configuration(nav_runtime):
    path = Path(__file__).resolve().parents[1] / 'config/recycle_tracking.yaml'
    config = yaml.safe_load(path.read_text())
    tracking = config['recycle_tracking_node']['ros__parameters']
    autonav = config['auto_nav']['ros__parameters']
    for name in ('scan_timeout_sec', 'future_stamp_tolerance_sec',
                 'clear_min_interval_sec', 'scan_restart_frames'):
        assert autonav['collision_' + name] == tracking['collision_' + name]
        assert getattr(nav_runtime.node._patrol_scan.config, name) == tracking['collision_' + name]
