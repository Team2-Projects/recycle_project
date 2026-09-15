"""Final-approach handoff using real methods and deterministic ROS transport doubles.

No physical motor, DDS, Nav2 planner or collision geometry is emulated here.
The existing adapter harness runs TrackingController/CollisionSafety themselves;
its results and health messages are passed to the actual AutoNav callbacks.
"""
from concurrent.futures import Future
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from test_autonav_recovery import nav_runtime, Handle, detection  # noqa: F401
from test_tracking_collision_adapter import runtime, Goal, Twist  # noqa: F401
from navigation.patrol_recovery import PatrolRecoveryConfig
from navigation.tracking_control import Command, Observation, Phase, TrackingConfig, TrackingController


def completed(message='FINAL_APPROACH_INTERRUPTED: collision', success=False, status=6):
    f = Future()
    f.set_result(NS(status=status, result=NS(success=success, message=message)))
    return f


def accept_tracking(h):
    n = h.node
    n.launch_recycle_tracking_action()
    generation = n._tracking_request_generation
    handle = Handle()
    n._recycle_tracking_client.calls[-1][1].set_result(handle)
    return generation, handle


def enable_final(h, duration=.8):
    h.node.config = replace(h.node.config, stop_only_test_mode=False,
                            final_approach_calibrated=True,
                            final_approach_duration_sec=duration)


def task_events(node):
    return [json.loads(m.data)['eventType'] for m in node.robot_task_pub.messages]


@pytest.mark.parametrize('suffix', [
    'collision slowing', 'cause=SAFE_STALE', 'cause=SCAN_STALE',
    'vision stream stopped', 'absolute approach timeout',
    'collision; CLEANUP_UNCONFIRMED: no response',
])
def test_final_interruption_retires_without_counting_and_returns_once(nav_runtime, suffix):
    h = nav_runtime
    n = h.node
    before_pan = len(n.pantilt_client.calls)
    n.recycle_tracking_result_callback(completed('FINAL_APPROACH_INTERRUPTED: ' + suffix))
    assert n._return_plan and n._return_plan.reason == 'FINAL_APPROACH_INTERRUPTED'
    assert n.check_timer is None and n.delay_timer is None
    assert n.collected_count == 0 and h.goal_count() == 0
    h.advance(count=16)
    assert h.goal_count() == 1 and n._nav_goal_pending
    assert len(n._recycle_tracking_client.calls) == 0  # no residual final retry
    assert len(n.pantilt_client.calls) == before_pan  # no false basket-check tilt
    assert n.collected_count == 0 and n._acquisition_lock.locked
    before = len(n.cmd_vel_pub.messages)
    h.advance(count=20)
    assert h.goal_count() == 1 and len(n.cmd_vel_pub.messages) == before
    assert 'OBJECT_PICKUP_FAIL' in task_events(n)


def test_interrupted_final_preserves_previously_collected_items(nav_runtime):
    h = nav_runtime
    n = h.node
    n.collected_count = 2
    n.previous_object_id = n.object_id = 2
    n.recycle_tracking_result_callback(completed())
    h.advance(count=15)
    assert n.collected_count == 2 and n.previous_object_id == 2 and n.object_id == 2
    assert h.goal_count() == 1


@pytest.mark.parametrize('block', ['ownership', 'motion', 'odom_missing'])
def test_do_not_close_or_navigate_until_tracking_released_and_stopped(nav_runtime, block):
    h = nav_runtime
    n = h.node
    count = len(n.servo_client.calls)
    if block == 'ownership':
        h.released = False
    elif block == 'motion':
        h.linear = .03
    else:
        h.odom_on = False
    n.recycle_tracking_result_callback(completed())
    h.advance(count=20)
    assert len(n.servo_client.calls) == count and h.goal_count() == 0
    h.released, h.linear, h.odom_on = True, 0., True
    h.advance(count=15)
    assert len(n.servo_client.calls) > count and h.goal_count() == 1


@pytest.mark.parametrize('block', ['scan', 'tf', 'nav2', 'servo', 'ownership'])
def test_final_recovery_does_not_require_manual_reset_after_real_inputs_recover(nav_runtime, block):
    h = nav_runtime
    n = h.node
    if block == 'scan':
        h.common = False
    elif block == 'tf':
        n._recovery_tf.good = False
    elif block == 'nav2':
        n._nav_state_clients['controller_server'].state = 2
    elif block == 'servo':
        n.servo_client.success = False
    else:
        h.released = False
    n.recycle_tracking_result_callback(completed())
    h.advance(count=25)
    assert h.goal_count() == 0 and n._return_plan is not None
    h.common = h.released = True
    n._recovery_tf.good = True
    n._nav_state_clients['controller_server'].state = 3
    n.servo_client.success = True
    h.advance(count=35)
    assert h.goal_count() == 1


def test_final_monitor_only_failure_allows_normal_nav_but_no_collection(nav_runtime):
    h = nav_runtime
    n = h.node
    h.collision = False
    n.recycle_tracking_result_callback(completed('FINAL_APPROACH_INTERRUPTED: cause=SAFE_STALE'))
    h.advance(count=15)
    assert h.goal_count() == 1
    handle = Handle()
    n._action_client.calls[-1][1].set_result(handle)
    n.object_callback(detection())
    assert not handle.cancels and len(n._recycle_tracking_client.calls) == 0


def test_final_servo_reply_timeout_then_recovery_keeps_pending_plan(nav_runtime):
    h = nav_runtime
    n = h.node
    n.servo_client.answer = False
    n.recycle_tracking_result_callback(completed())
    h.advance(count=40)
    assert h.goal_count() == 0
    assert 1 < len(n.servo_client.calls) < 7
    n.servo_client.answer = True
    h.advance(count=35)
    assert h.goal_count() == 1


def test_final_auto_return_can_be_disabled_without_changing_other_failures(nav_runtime):
    h = nav_runtime
    n = h.node
    n._recovery_cfg = replace(n._recovery_cfg, final_interrupted_enabled=False)
    n.recycle_tracking_result_callback(completed())
    h.advance(count=20)
    assert h.goal_count() == 0 and n._return_plan is None
    # The explicit operator path remains available, with the same checks.
    assert n.reset_tracking_hold_callback(None, NS()).success
    h.advance(count=15)
    assert h.goal_count() == 1


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_final_policy_switch_must_be_bool(value):
    with pytest.raises(ValueError):
        replace(PatrolRecoveryConfig(), final_interrupted_enabled=value)


@pytest.mark.parametrize('reason', ['TEST_STOP: near', 'INTERNAL_ERROR: bad',
                                   'SAFETY_INVALID_OUTPUT: nan', 'SERVICE_ERROR: fail'])
def test_final_return_does_not_make_unknown_or_test_results_recoverable(nav_runtime, reason):
    h = nav_runtime
    h.node.recycle_tracking_result_callback(completed(reason))
    h.advance(count=25)
    assert h.goal_count() == 0 and h.node._tracking_safety_hold


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
def test_operator_after_final_failure_cancels_delayed_servo_and_resume(nav_runtime, command):
    h = nav_runtime
    n = h.node
    n.servo_client.answer = False
    n.recycle_tracking_result_callback(completed())
    h.advance(count=3)
    old_future = n._close_future
    assert old_future is not None
    n.command_callback(NS(data=command))
    assert n._return_plan.destination == 'HOME'
    # Local waiter cancelled. Hardware call may complete later but it is not
    # permission to create a patrol goal. No forced/implicit reset is issued.
    if not old_future.done():
        old_future.set_result(NS(success=True))
    n.servo_client.answer = True
    h.advance(count=30)
    assert h.goal_count() == 1
    assert n._action_client.calls[-1][0].pose.pose.position.x == n.home_x


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
def test_operator_before_final_result_prevents_auto_patrol(nav_runtime, command):
    h = nav_runtime
    n = h.node
    _, handle = accept_tracking(h)
    n.command_callback(NS(data=command))
    assert handle.cancels == 1
    handle.result.set_result(completed().result())
    h.advance(count=15)
    assert h.goal_count() == 1 and n._return_plan is None
    assert n._action_client.calls[-1][0].pose.pose.position.x == n.home_x


def test_stop_after_final_return_request_cancels_late_nav_acceptance(nav_runtime):
    h = nav_runtime
    n = h.node
    n.recycle_tracking_result_callback(completed())
    h.advance(count=12)
    n.command_callback(NS(data='STOP'))
    goal = Handle()
    n._action_client.calls[-1][1].set_result(goal)
    assert goal.cancels == 1
    before = h.goal_count()
    h.advance(count=20)
    assert h.goal_count() == before and n._return_plan.destination == 'HOME'
    goal.result.set_result(completed('STOP', False, 5).result())
    h.advance(count=15)
    assert h.goal_count() == before + 1
    assert n._action_client.calls[-1][0].pose.pose.position.x == n.home_x


def test_duplicate_interrupted_result_does_not_stop_new_nav_patrol(nav_runtime):
    h = nav_runtime
    n = h.node
    generation, handle = accept_tracking(h)
    handle.result.set_result(completed().result())
    h.advance(count=15)
    assert h.goal_count() == 1
    before = len(n.cmd_vel_pub.messages)
    n.recycle_tracking_result_callback(handle.result, generation)
    h.advance(count=10)
    assert h.goal_count() == 1 and len(n.cmd_vel_pub.messages) == before


def test_duplicate_success_counts_once_and_has_one_basket_timer(nav_runtime):
    h = nav_runtime
    n = h.node
    generation, handle = accept_tracking(h)
    handle.result.set_result(completed('SUCCESS: done', True, 4).result())
    timer = n.check_timer
    before_timers = len(n.timers)
    n.recycle_tracking_result_callback(handle.result, generation)
    assert n.collected_count == 1 and n.check_timer is timer
    assert len(n.timers) == before_timers


def test_old_result_cannot_clear_a_new_tracking_handle(nav_runtime):
    h = nav_runtime
    n = h.node
    generation, old_handle = accept_tracking(h)
    old_handle.result.set_result(completed().result())
    n._cancel_auto_return('TEST_NEW_REQUEST')
    _, current_handle = accept_tracking(h)
    n.recycle_tracking_result_callback(old_handle.result, generation)
    assert n.tracking_handle is current_handle and n._return_plan is None


def test_late_old_goal_acceptance_is_cancelled_not_adopted(nav_runtime):
    h = nav_runtime
    n = h.node
    n._tracking_request_generation = 2
    n._tracking_goal_pending = True
    old = Handle()
    f = Future(); f.set_result(old)
    n.recycle_tracking_goal_response_callback(f, generation=1)
    assert old.cancels == 1 and n.tracking_handle is None and n._tracking_goal_pending


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
@pytest.mark.parametrize('stage', ['basket_timer', 'resume_timer'])
def test_success_timer_cannot_restart_patrol_after_stop(nav_runtime, command, stage):
    h = nav_runtime
    n = h.node
    n.recycle_tracking_result_callback(completed('SUCCESS: done', True, 4))
    old_timer = n.check_timer
    if stage == 'resume_timer':
        old_timer.callback()
        old_timer = n.delay_timer
    assert old_timer is not None
    n.command_callback(NS(data=command))
    old_timer.callback()  # emulate a timer already queued before cancel()
    assert h.goal_count() == 0 and not n._recycle_client.calls


def test_old_success_timer_does_not_cancel_new_return_plan(nav_runtime):
    h = nav_runtime
    n = h.node
    n.recycle_tracking_result_callback(completed('SUCCESS: old', True, 4))
    old = n.check_timer
    n._handle_tracking_failure('FINAL_APPROACH_INTERRUPTED: later attempt')
    plan = n._return_plan
    old.callback()
    assert n._return_plan is plan and h.goal_count() == 0
    h.advance(count=15)
    assert h.goal_count() == 1


def test_normal_success_retains_existing_basket_and_delayed_resume_flow(nav_runtime):
    h = nav_runtime
    n = h.node
    n.recycle_tracking_result_callback(completed('SUCCESS: done', True, 4))
    assert n.collected_count == 1 and n._return_plan is None
    assert n.pantilt_client.calls[-1][0].angle == 90.
    assert n.check_timer is not None
    h.now += 3.
    n.y_min = None
    n.check_timer.callback()
    assert n.pantilt_client.calls[-1][0].angle == 151.
    assert n.delay_timer is not None
    h.now += 2.
    n.delay_timer.callback()
    assert h.goal_count() == 1 and not n._recycle_client.calls


def test_normal_success_full_basket_keeps_existing_disposal_path(nav_runtime):
    h = nav_runtime
    n = h.node
    n.recycle_tracking_result_callback(completed('SUCCESS: done', True, 4))
    n.y_min = 100.
    n.check_timer.callback()
    assert n.collected_count == 1 and len(n._recycle_client.calls) == 1
    assert h.goal_count() == 0


def test_final_hard_cap_is_interruption_not_success_or_generic_timeout():
    cfg = TrackingConfig(stop_only_test_mode=False, final_approach_calibrated=True)
    c = TrackingController(cfg, 0, 0., defer_final_motion=True)
    c.phase = Phase.FINAL_APPROACH
    c.approach_started_at = 0.
    c.latest = Observation(1, 59.9, 0, .9, 350., 405., 40., 50.)
    c.last_sequence = 1
    c.arm_final_motion(59.5)
    assert c.step(60.) == Command()
    assert c.reason.startswith('FINAL_APPROACH_INTERRUPTED:') and c.phase == Phase.FAILED


@pytest.mark.parametrize('ratio', [0., .95])
def test_actual_final_collision_interrupts_before_success(runtime, ratio):
    h = runtime
    enable_final(h)
    h.ratio = lambda t, raw: ratio if h.node._safety.final_armed else 1.
    result = h.node.execute_callback(h.goal)
    assert not result.success and result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert h.goal.state == 'ABORTED' and not h.node._owns_cmd_vel
    assert h.node.cmd_vel_pub.messages[-1].linear.x == 0.


@pytest.mark.parametrize('fault', ['scan', 'monitor', 'vision'])
def test_actual_final_stream_loss_aborts_attempt_no_auto_final_resumption(runtime, fault):
    h = runtime
    enable_final(h, duration=5.)
    old_advance = h.advance
    def advance(dt):
        if h.node._safety.final_armed:
            setattr(h, dict(scan='scan_on', monitor='monitor_on', vision='vision_on')[fault], False)
        old_advance(dt)
    h.advance = advance
    result = h.node.execute_callback(h.goal)
    assert not result.success and result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert h.node.cmd_vel_pub.messages[-1].linear.x == 0.


@pytest.mark.parametrize('terminal', ['success', 'interrupted'])
def test_cleanup_period_and_late_safe_never_change_committed_outcome(runtime, terminal):
    h = runtime
    n = h.node
    enable_final(h)
    if terminal == 'interrupted':
        h.ratio = lambda t, raw: .5 if n._safety.final_armed else 1.
    cleanup_snapshot = []
    def srv(enable, *args, **kwargs):
        if not enable:
            cleanup_snapshot.append(len(n.cmd_vel_pub.messages))
            assert n._control_finished and n._controller is None
            for _ in range(20):
                h.advance(.1)
                n._safe_callback(n._twist(Command(.03)))
                n._publish_command(Command(.03))
                n._collision_watchdog()
        return True, 'OK'
    n.call_tracking_srv = srv
    result = n.execute_callback(h.goal)
    assert result.success is (terminal == 'success')
    assert result.message.startswith('SUCCESS:' if terminal == 'success' else 'FINAL_APPROACH_INTERRUPTED:')
    for m in n.cmd_vel_pub.messages[cleanup_snapshot[0]:]:
        assert m.linear.x == 0 and m.angular.z == 0
    count = len(n.cmd_vel_pub.messages)
    for _ in range(5):
        h.advance(.1)
        n._safe_callback(n._twist(Command(.03)))
        n._publish_health()
    assert len(n.cmd_vel_pub.messages) == count
    assert json.loads(n._health_pub.messages[-1].data)['released']


def test_cancel_during_cleanup_wins_over_completed_final_motion(runtime):
    h = runtime
    n = h.node
    enable_final(h)
    def srv(enable, *args, **kwargs):
        if not enable:
            h.goal.is_cancel_requested = True
            n.cancel_callback(h.goal)
        return True, 'OK'
    n.call_tracking_srv = srv
    result = n.execute_callback(h.goal)
    assert not result.success and result.message == 'STOP' and h.goal.state == 'CANCELED'


def test_failed_cleanup_preserves_primary_final_reason(runtime):
    h = runtime
    n = h.node
    enable_final(h)
    h.ratio = lambda t, raw: .5 if n._safety.final_armed else 1.
    n.call_tracking_srv = lambda enable, *a, **kw: (True, 'OK') if enable else (False, 'SERVICE_TIMEOUT')
    result = n.execute_callback(h.goal)
    assert result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert 'CLEANUP_UNCONFIRMED' in result.message
    assert not result.success and not n._owns_cmd_vel


def test_final_cleanup_clock_order_and_new_action_reset(runtime):
    h = runtime
    n = h.node
    enable_final(h)
    def advancing_clock():
        h.now += .0001
        return h.now
    h.mod.time = NS(monotonic=advancing_clock)
    first = n.execute_callback(h.goal)
    assert first.success
    assert n._control_finished and not n._goal_busy and not n._owns_cmd_vel
    h.goal = Goal()
    assert n.goal_callback(h.goal.request) == 1
    second = n.execute_callback(h.goal)
    assert second.success and not n._owns_cmd_vel


@pytest.mark.parametrize('fault', ['collision', 'scan', 'monitor'])
def test_actual_adapter_result_health_and_autonav_handoff(nav_runtime, runtime, fault):
    a, t = nav_runtime, runtime
    n, tracking = a.node, t.node
    enable_final(t, 5.)
    if fault == 'collision':
        t.ratio = lambda at, raw: .4 if tracking._safety.final_armed else 1.
    else:
        original_advance = t.advance
        def advance(dt):
            if tracking._safety.final_armed:
                setattr(t, 'scan_on' if fault == 'scan' else 'monitor_on', False)
            original_advance(dt)
        t.advance = advance
    result = tracking.execute_callback(t.goal)
    assert result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert not result.success and not tracking._owns_cmd_vel and not tracking._goal_busy
    a.now = max(a.now, t.now) + .1
    f = Future(); f.set_result(NS(status=6, result=result))
    n.recycle_tracking_result_callback(f)
    assert n.collected_count == 0
    a.health_on = False  # no invented ready report: use actual adapter output
    a.scan_on = False
    motor_count = len(tracking.cmd_vel_pub.messages)
    for i in range(60):
        t.now = a.now
        if i >= 20:
            t.scan_on = t.monitor_on = True
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
            assert a.goal_count() == (0 if fault == 'scan' else 1)
    assert a.goal_count() == 1 and n._acquisition_lock.locked
    assert n.collected_count == 0 and not n._recycle_tracking_client.calls
    assert len(tracking.cmd_vel_pub.messages) == motor_count


def test_production_yaml_runs_full_ten_seconds_from_first_approved_command(runtime):
    from dataclasses import fields
    from pathlib import Path
    import yaml
    h = runtime
    n = h.node
    p = Path(__file__).resolve().parents[1] / 'config/recycle_tracking.yaml'
    params = yaml.safe_load(p.read_text())['recycle_tracking_node']['ros__parameters']
    n.config = TrackingConfig(**{f.name: params[f.name] for f in fields(TrackingConfig())})
    assert not n.config.stop_only_test_mode and n.config.final_approach_calibrated
    assert n.config.final_approach_duration_sec == 10.
    first = []
    publish = n.cmd_vel_pub.publish
    def record(msg):
        if (not first and n._controller is not None
                and n._controller.phase == Phase.FINAL_APPROACH and msg.linear.x > 0):
            first.append(h.now)
        publish(msg)
    n.cmd_vel_pub.publish = record
    result = n.execute_callback(h.goal)
    assert result.success and first and h.now - first[0] >= 10.
    assert not n._owns_cmd_vel and n.cmd_vel_pub.messages[-1].linear.x == 0.


def test_intervention_just_before_final_deadline_is_not_overwritten_by_success(runtime):
    h = runtime
    n = h.node
    enable_final(h, duration=10.)
    def ratio(t, raw):
        c = n._controller
        if c is not None and n._safety.final_armed and t - c.final_started_at >= 9.8:
            return .9
        return 1.
    h.ratio = ratio
    result = n.execute_callback(h.goal)
    assert not result.success and result.message.startswith('FINAL_APPROACH_INTERRUPTED:')
    assert h.goal.state == 'ABORTED'
