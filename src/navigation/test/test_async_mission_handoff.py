"""Mission transitions with real methods and delayed transport boundaries."""
from concurrent.futures import Future
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from test_autonav_recovery import nav_runtime, Handle, detection  # noqa: F401
from test_tracking_collision_adapter import runtime, Goal  # noqa: F401


def outcome(status=6, message='FINAL_APPROACH_INTERRUPTED: test', success=False):
    return NS(status=status, result=NS(success=success, message=message))


def completed(row):
    future = Future()
    future.set_result(row)
    return future


def accept_tracking(h):
    h.node.launch_recycle_tracking_action()
    handle = Handle()
    h.node._recycle_tracking_client.calls[-1][1].set_result(handle)
    return handle


def assert_home(h, count=1):
    assert h.goal_count() == count
    goal = h.node._action_client.calls[-1][0].pose.pose.position
    assert (goal.x, goal.y) == (h.node.home_x, h.node.home_y)
    assert h.node._nav_goal_purpose == 'HOME'


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
@pytest.mark.parametrize('block', ['release', 'motion', 'servo', 'common', 'odom', 'nav2'])
def test_tracking_stop_waits_then_sends_home_once(nav_runtime, command, block):
    h, n = nav_runtime, nav_runtime.node
    handle = accept_tracking(h)
    n.command_callback(NS(data=command))
    assert handle.cancels == 1 and n._return_plan.destination == 'HOME'
    if block == 'release':
        h.released = False
    elif block == 'motion':
        h.linear = .05
    elif block == 'servo':
        n.servo_client.answer = False
    elif block == 'common':
        h.common = False
    elif block == 'odom':
        h.odom_on = False
    else:
        n._action_client.ready = False
    # Neither monitor-only faults nor unfinished perception cleanup gate HOME.
    h.collision = h.cleanup = False
    terminal = outcome(message='FINAL_APPROACH_INTERRUPTED: test; CLEANUP_UNCONFIRMED: timeout')
    if block != 'release':
        handle.result.set_result(terminal)
    h.advance(count=6)
    assert h.goal_count() == 0
    h.common, h.odom_on, h.linear = True, True, 0.
    if block == 'release':
        handle.result.set_result(terminal)
    n._action_client.ready = True
    if block == 'servo':
        pending_close = n._close_future
        assert pending_close is not None and not pending_close.done()
        pending_close.set_result(NS(success=True))
    h.advance(count=8)
    assert_home(h)
    assert 'set_tracking_mode' not in n.clients
    h.advance(count=10)
    assert_home(h)


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
@pytest.mark.parametrize('status', [4, 5, 6])
@pytest.mark.parametrize('pending', [False, True])
def test_old_patrol_result_cannot_override_home(nav_runtime, command, status, pending):
    h, n = nav_runtime, nav_runtime.node
    n.object_found = False
    n.send_next_goal()
    old_generation = n._nav_request_generation
    acceptance = n._action_client.calls[-1][1]
    old = Handle()
    if not pending:
        acceptance.set_result(old)
    n.command_callback(NS(data=command))
    if pending:
        h.advance(count=5)
        assert h.goal_count() == 1
        acceptance.set_result(old)
    assert old.cancels == 1
    old.result.set_result(outcome(status))
    h.advance(count=10)
    assert_home(h, 2)
    assert n.current_idx == 0
    home = Handle()
    home_acceptance = n._action_client.calls[-1][1]
    home_acceptance.set_result(home)
    home_generation = n._nav_request_generation
    # Late/duplicate callbacks must not clear the HOME handle or advance patrol.
    n.result_callback(old.result, old_generation, 'PATROL')
    n.goal_response_callback(acceptance, old_generation, 'PATROL')
    n.goal_response_callback(home_acceptance, home_generation, 'HOME')
    assert n.current_handle is home and home.cancels == 0
    n.command_callback(NS(data=command))
    assert home.cancels == 0
    home.result.set_result(outcome(4))
    n.result_callback(home.result, home_generation, 'HOME')
    assert n._home_completed and not n.is_running
    assert n.current_idx == 0
    assert_home(h, 2)


@pytest.mark.parametrize('kind', ['navigation', 'tracking', 'recycle'])
def test_stop_waiting_acceptance_handles_rejection(nav_runtime, kind):
    h, n = nav_runtime, nav_runtime.node
    if kind == 'navigation':
        n.send_next_goal()
        client = n._action_client
    elif kind == 'tracking':
        n.launch_recycle_tracking_action()
        client = n._recycle_tracking_client
    else:
        n.launch_recycle_action()
        client = n._recycle_client
    response = client.calls[-1][1]
    n.command_callback(NS(data='STOP'))
    h.advance(count=6)
    before = h.goal_count()
    response.set_result(Handle(accepted=False))
    h.advance(count=10)
    assert_home(h, before + 1)
    assert not n.stop_pending and n.current_idx == 0


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
@pytest.mark.parametrize('stage', ['basket', 'resume', 'idle'])
def test_stop_without_action_does_not_wait_for_nonexistent_callback(nav_runtime, command, stage):
    h, n = nav_runtime, nav_runtime.node
    old_timer = None
    if stage != 'idle':
        handle = accept_tracking(h)
        handle.result.set_result(outcome(4, 'SUCCESS: done', True))
        old_timer = n.check_timer
        if stage == 'resume':
            old_timer.callback()
            old_timer = n.delay_timer
    n.command_callback(NS(data=command))
    assert not n.stop_pending and n._return_plan.destination == 'HOME'
    if old_timer:
        old_timer.callback()
    h.advance(count=12)
    assert_home(h)
    assert n.check_timer is None and n.delay_timer is None
    assert n.collected_count == (stage != 'idle')


@pytest.mark.parametrize('status', [4, 5, 6])
def test_recycle_result_after_stop_does_not_restart_patrol_or_reset_count(nav_runtime, status):
    h, n = nav_runtime, nav_runtime.node
    n.collected_count = 2
    n.launch_recycle_action()
    generation = n._recycle_request_generation
    handle = Handle()
    acceptance = n._recycle_client.calls[-1][1]
    n.command_callback(NS(data='STOP'))
    acceptance.set_result(handle)
    assert handle.cancels == 1
    handle.result.set_result(outcome(status, 'done', status == 4))
    h.advance(count=10)
    n.recycle_result_callback(handle.result, generation)
    n.recycle_goal_response_callback(acceptance, generation)
    assert_home(h)
    assert n.collected_count == 2 and n.current_idx == 0 and n.recycle_handle is None


def test_home_abort_retries_home_with_existing_limit(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    n.command_callback(NS(data='STOP'))
    for i in range(n.max_abort_retry):
        h.advance(count=10)
        assert_home(h, i + 1)
        handle = Handle()
        n._action_client.calls[-1][1].set_result(handle)
        handle.result.set_result(outcome(6))
    h.advance(count=15)
    assert_home(h, n.max_abort_retry)
    assert n.current_idx == 0 and not n._home_completed


def test_normal_patrol_and_collection_keep_count_timers_and_index(nav_runtime):
    h, n = nav_runtime, nav_runtime.node
    n.object_found = False
    n.send_next_goal()
    first = Handle()
    n._action_client.calls[-1][1].set_result(first)
    first.result.set_result(outcome(4))
    assert n.current_idx == 1 and h.goal_count() == 2
    patrol = Handle()
    n._action_client.calls[-1][1].set_result(patrol)
    h.advance(count=5)
    n.object_callback(detection())
    assert patrol.cancels == 1
    patrol.result.set_result(outcome(5))
    tracking = Handle()
    acceptance = n._recycle_tracking_client.calls[-1][1]
    acceptance.set_result(tracking)
    generation = n._tracking_request_generation
    tracking.result.set_result(outcome(4, 'SUCCESS: done', True))
    n.recycle_tracking_result_callback(tracking.result, generation)
    n.recycle_tracking_goal_response_callback(acceptance, generation)
    assert n.collected_count == 1 and n.tracking_handle is None
    assert n.servo_client.calls[-1][0].angle1 == 0.
    assert n.pantilt_client.calls[-1][0].angle == 90.
    n.check_timer.callback()
    n.delay_timer.callback()
    assert h.goal_count() == 3 and n.current_idx == 1
    assert n._action_client.calls[-1][0].pose.pose.position.x == n.waypoints[1][0]


@pytest.mark.parametrize('collected', [0, 2])
@pytest.mark.parametrize('retry', [False, True])
def test_final_waypoint_ignores_detections_and_keeps_completion_flow(nav_runtime, collected, retry):
    h, n = nav_runtime, nav_runtime.node
    n.object_found = False
    n.collected_count = collected
    n.current_idx = len(n.waypoints) - 1
    n.send_next_goal()
    home = Handle()
    n._action_client.calls[-1][1].set_result(home)
    if retry:
        home.result.set_result(outcome(6))
        home = Handle()
        n._action_client.calls[-1][1].set_result(home)
    assert n._action_client.calls[-1][0].pose.pose.position.x == n.home_x
    servo_calls = len(n.servo_client.calls)
    n.object_callback(detection())
    assert home.cancels == 0
    assert not n._recycle_tracking_client.calls
    assert len(n.servo_client.calls) == servo_calls
    assert n.collected_count == collected
    home.result.set_result(outcome(4))
    assert n.current_idx == len(n.waypoints)
    assert len(n._recycle_client.calls) == int(collected > 0)
    if not collected:
        assert json.loads(n.schedule_status_pub.messages[-1].data)['status'] == 'COMPLETE'


@pytest.fixture
def mode_runtime(runtime, monkeypatch):
    """Use production service sequencing; fake only clock/wait and ROS client."""
    h, n = runtime, runtime.node
    del n.call_tracking_srv  # remove the older harness's transport shortcut
    h.mode_policy = lambda enable, number: (0., True, 'OK')

    class Client:
        def __init__(self):
            self.ready = True
            self.calls = []

        def service_is_ready(self):
            return self.ready

        def call_async(self, request):
            future = Future()
            delay, success, reason = h.mode_policy(request.enable, len(self.calls))
            self.calls.append(NS(request=request, future=future,
                                 due=None if delay is None else h.now + delay,
                                 success=success, reason=reason))
            return future

    n.tracking_cli = Client()

    def deliver():
        for call in n.tracking_cli.calls:
            if call.due is not None and h.now >= call.due and not call.future.done():
                call.future.set_result(NS(success=call.success, reason=call.reason))

    h.env_hook = deliver

    class WaitEvent:
        def __init__(self):
            self.value = False

        def set(self):
            self.value = True

        def wait(self, timeout):
            if not self.value:
                h.advance(timeout)
            return self.value

    monkeypatch.setattr(h.mod, 'Event', WaitEvent)
    return h


def test_timed_out_off_remains_owned_until_late_response(mode_runtime):
    h, n = mode_runtime, mode_runtime.node
    h.mode_policy = lambda enable, number: (None, True, 'OK')
    assert n.call_tracking_srv(False) == (False, 'SERVICE_TIMEOUT')
    call = n.tracking_cli.calls[0]
    assert n._mode_future is call.future and not call.future.cancelled()
    for _ in range(50):
        h.advance(.1)
        n._cleanup_tick()
    assert len(n.tracking_cli.calls) == 1  # no later OFF may overtake this one
    assert n.goal_callback(Goal().request) == h.mod.GoalResponse.REJECT
    assert n.call_tracking_srv(True, 0) == (False, 'CLEANUP_PENDING')
    call.future.set_result(NS(success=True, reason='OK'))
    n._cleanup_tick()
    assert n._mode_cleanup_ready
    h.mode_policy = lambda enable, number: (0., True, 'OK')
    assert n.call_tracking_srv(True, 0) == (True, 'OK')
    assert [c.request.enable for c in n.tracking_cli.calls] == [False, True]
    assert not n._mode_cleanup_ready


def test_late_on_must_finish_before_off_can_be_sent(mode_runtime):
    h, n = mode_runtime, mode_runtime.node
    h.mode_policy = lambda enable, number: (None, True, 'OK')
    assert n.call_tracking_srv(True, 0)[1] == 'SERVICE_TIMEOUT'
    pending = n._mode_future
    assert n.call_tracking_srv(False)[1] == 'SERVICE_TIMEOUT'
    assert len(n.tracking_cli.calls) == 1 and n._mode_future is pending
    pending.set_result(NS(success=True, reason='OK'))
    h.mode_policy = lambda enable, number: (0., True, 'OK')
    n._cleanup_tick()
    assert [c.request.enable for c in n.tracking_cli.calls] == [True, False]
    h.advance(.1)
    n._cleanup_tick()
    assert n._mode_cleanup_ready


def test_failed_off_response_retries_without_new_on(mode_runtime):
    h, n = mode_runtime, mode_runtime.node
    h.mode_policy = lambda enable, number: (0., number > 0, 'OK' if number else 'INTERNAL_ERROR')
    assert n.call_tracking_srv(False) == (False, 'INTERNAL_ERROR')
    assert not n._mode_cleanup_ready
    for _ in range(40):
        h.advance(.1)
        n._cleanup_tick()
    assert n._mode_cleanup_ready
    assert all(not c.request.enable for c in n.tracking_cli.calls)


def test_cancel_during_pending_on_retains_request_until_off_settles(mode_runtime):
    h, n = mode_runtime, mode_runtime.node
    h.mode_policy = lambda enable, number: (None, True, 'OK')
    deliver = h.env_hook

    def cancel_after_request():
        deliver()
        if not h.goal.is_cancel_requested:
            h.goal.is_cancel_requested = True
            n.cancel_callback(h.goal)

    h.env_hook = cancel_after_request
    assert n.goal_callback(h.goal.request) == h.mod.GoalResponse.ACCEPT
    result = n.execute_callback(h.goal)
    assert not result.success and result.message.startswith('STOP; CLEANUP_UNCONFIRMED:')
    assert h.goal.state == 'CANCELED' and not n._owns_cmd_vel
    pending = n._mode_future
    assert pending is n.tracking_cli.calls[0].future and not pending.cancelled()
    assert n.goal_callback(Goal().request) == h.mod.GoalResponse.REJECT
    pending.set_result(NS(success=True, reason='OK'))
    h.mode_policy = lambda enable, number: (0., True, 'OK')
    n._cleanup_tick()
    assert [c.request.enable for c in n.tracking_cli.calls] == [True, False]
    h.advance(.1)
    n._cleanup_tick()
    assert n.goal_callback(Goal().request) == h.mod.GoalResponse.ACCEPT
    assert n.call_tracking_srv(True, 0) == (True, 'OK')
    assert [c.request.enable for c in n.tracking_cli.calls] == [True, False, True]


@pytest.mark.parametrize('primary', ['LOST_TARGET', 'VISION_NOT_READY'])
def test_primary_failure_with_pending_cleanup_can_resume_patrol(nav_runtime, mode_runtime, primary):
    a, t = nav_runtime, mode_runtime
    n, tracker = a.node, t.node
    t.mode_policy = lambda enable, number: (
        (0., primary != 'VISION_NOT_READY', 'OK' if primary != 'VISION_NOT_READY' else primary)
        if enable else (None, True, 'OK'))
    if primary == 'LOST_TARGET':
        t.position = lambda at: (float('nan'), 300.)
    result = tracker.execute_callback(t.goal)
    assert not result.success and result.message.startswith(primary + ':')
    assert 'CLEANUP_UNCONFIRMED' in result.message
    assert not tracker._owns_cmd_vel and tracker._mode_future is not None
    n.recycle_tracking_result_callback(completed(NS(status=6, result=result)))
    a.health_on = False
    a.now = max(a.now, t.now) + .1
    t.monitor_on = False  # monitor-only failure does not block Nav2 patrol
    for _ in range(30):
        t.now = a.now
        t.environment()
        tracker._cleanup_tick()
        a.advance(.1)
    assert a.goal_count() == 1 and not tracker._mode_cleanup_ready
    assert n._tracking_hold_reason == '' and n._acquisition_lock.locked
    assert n.collected_count == 0


@pytest.mark.parametrize('cleanup_first', [False, True])
def test_patrol_rearm_and_tracking_off_have_separate_owners(nav_runtime, mode_runtime, cleanup_first):
    h, n = nav_runtime, nav_runtime.node
    t, tracking = mode_runtime, mode_runtime.node
    t.mode_policy = lambda enable, number: (None, True, 'OK')
    tracking.call_tracking_srv(False)
    pending = tracking._mode_future
    n._handle_tracking_failure('LOST_TARGET: gone; CLEANUP_UNCONFIRMED: timeout')
    h.advance(count=5)
    patrol = Handle()
    n._action_client.calls[-1][1].set_result(patrol)
    if cleanup_first:
        pending.set_result(NS(success=True, reason='OK'))
        tracking._cleanup_tick()
    n.object_callback(detection())
    assert patrol.cancels == 0 and n._acquisition_lock.locked
    for i in range(1, 13):
        h.x = i * .01
        h.advance(.1)
    patrol.result.set_result(outcome(4))
    assert not n._acquisition_lock.locked
    next_patrol = Handle()
    n._action_client.calls[-1][1].set_result(next_patrol)
    h.advance(count=5)  # Existing acquisition cooldown also remains in force.
    n.object_callback(detection())
    next_patrol.result.set_result(outcome(5))
    assert len(n._recycle_tracking_client.calls) == 1
    allowed = tracking.goal_callback(Goal().request)
    assert allowed == (1 if cleanup_first else 0)
    if not cleanup_first:
        n._recycle_tracking_client.calls[-1][1].set_result(Handle(accepted=False))
        assert n._return_plan is not None and n._acquisition_lock.locked
        assert not any(c.request.enable for c in tracking.tracking_cli.calls)
        pending.set_result(NS(success=True, reason='OK'))
        tracking._cleanup_tick()
        assert tracking.goal_callback(Goal().request) == 1


def test_manual_retry_is_rejected_by_tracking_while_off_is_pending(nav_runtime, mode_runtime):
    h, n = nav_runtime, nav_runtime.node
    t, tracking = mode_runtime, mode_runtime.node
    t.mode_policy = lambda enable, number: (None, True, 'OK')
    tracking.call_tracking_srv(False)
    h.common = False
    n._handle_tracking_failure('SENSOR_STALE: gap; CLEANUP_UNCONFIRMED: timeout')
    h.advance(count=6)
    response = n.resume_sensor_hold_callback(None, NS())
    assert response.success and n._tracking_goal_pending
    assert tracking.goal_callback(Goal().request) == 0
    n._recycle_tracking_client.calls[-1][1].set_result(Handle(accepted=False))
    assert n._return_plan is not None
    assert not any(c.request.enable for c in tracking.tracking_cli.calls)


@pytest.mark.parametrize('terminal', ['success', 'test_stop', 'interrupted'])
def test_cleanup_failure_keeps_primary_outcome_and_count(mode_runtime, nav_runtime, terminal):
    t, a = mode_runtime, nav_runtime
    # Runtime modules are independent; the service client stays bound to its node.
    if terminal != 'test_stop':
        t.node.config = replace(t.node.config, stop_only_test_mode=False,
                                final_approach_calibrated=True, final_approach_duration_sec=.8)
    if terminal == 'interrupted':
        t.ratio = lambda at, raw: .5 if raw.linear.x == .03 else 1.
    t.mode_policy = lambda enable, number: (0., True, 'OK') if enable else (None, True, 'OK')
    result = t.node.execute_callback(t.goal)
    code = {'success': 'SUCCESS', 'test_stop': 'TEST_STOP',
            'interrupted': 'FINAL_APPROACH_INTERRUPTED'}[terminal]
    assert result.message.startswith(code + ':') and 'CLEANUP_UNCONFIRMED' in result.message
    assert result.success == (terminal == 'success')
    handle = accept_tracking(a)
    generation = a.node._tracking_request_generation
    handle.result.set_result(NS(status=4 if result.success else 6, result=result))
    a.node.recycle_tracking_result_callback(handle.result, generation)
    assert a.node.collected_count == (terminal == 'success')
    assert not t.node._mode_cleanup_ready
    assert (a.node.check_timer is not None) == (terminal == 'success')
