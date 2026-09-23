"""이동 요청 수락·종료와 검출·STOP의 순서를 바꾸어 수거 연결을 검증한다."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from builtin_interfaces.msg import Time
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from navigation import auto_nav

import pytest


class ActionClient:
    """요청을 기록하고 응답 시점을 테스트에서 제어한다."""

    def __init__(self):
        """전송한 목표와 수락 응답 Future를 저장한다."""
        self.requests = []

    def wait_for_server(self):
        """테스트용 Action 서버는 즉시 사용 가능하다."""
        return True

    def send_goal_async(self, goal, **kwargs):
        """나중에 수락하거나 거절할 목표를 기록한다."""
        future = Future()
        self.requests.append((goal, future))
        return future


class GoalHandle:
    """취소 요청과 최종 이동·수거 결과를 따로 제어한다."""

    def __init__(self):
        """수락된 목표와 아직 완료되지 않은 결과를 만든다."""
        self.accepted = True
        self.result = Future()
        self.cancel_goal_async = Mock(side_effect=lambda: Future())

    def get_result_async(self):
        """목표의 최종 결과를 기다리는 Future를 반환한다."""
        return self.result

    def finish(self, status, success=False):
        """서버가 보낸 최종 상태로 실제 노드 콜백을 실행한다."""
        self.result.set_result(SimpleNamespace(
            status=status, result=SimpleNamespace(success=success, message='test result')))


class InferenceService:
    """정상 응답과 지연·유실 응답을 전환하는 추론 제어 서비스."""

    def __init__(self):
        self.requests = []
        self.available = True
        self.auto_respond = True
        self.remove_pending_request = Mock()

    def service_is_ready(self):
        return self.available

    def call_async(self, request):
        future = Future()
        self.requests.append((request.data, future))
        if self.auto_respond:
            future.set_result(SimpleNamespace(success=True, message='ok'))
        return future


def detection(x=340.0, class_id=0):
    """유효한 물체 또는 미검출 관측을 만든다."""
    return SimpleNamespace(id=class_id, coord=[x, 300.0, 60.0, 80.0],
                           confidence=0.9, min_y=260.0)


def accept(client):
    """가장 최근 요청을 수락해 응답 콜백을 실행한다."""
    handle = GoalHandle()
    client.requests[-1][1].set_result(handle)
    return handle


@pytest.fixture
def nav_rig(monkeypatch):
    """실제 AutoNav를 초기화하고 ROS 입출력과 경과 시간만 대체한다."""
    clients, publishers, services = {}, {}, {}
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(auto_nav, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(auto_nav.Node, '__init__', lambda self, name: None)
    monkeypatch.setattr(auto_nav.Node, 'declare_parameter',
                        lambda self, name, default: SimpleNamespace(value=default))
    monkeypatch.setattr(auto_nav.Node, 'get_logger', lambda self: Mock())
    monkeypatch.setattr(auto_nav.Node, 'get_clock', lambda self: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(), nanoseconds=int((100 + clock.now) * 1e9))))
    monkeypatch.setattr(auto_nav.Node, 'create_subscription', lambda *a, **kw: None)
    monkeypatch.setattr(auto_nav.Node, 'create_timer', lambda *a: Mock())
    monkeypatch.setattr(auto_nav.Node, 'destroy_timer', lambda *a: None)
    monkeypatch.setattr(auto_nav.Node, 'create_publisher',
                        lambda self, msg, topic, qos: publishers.setdefault(topic, Mock()))
    inference = InferenceService()
    monkeypatch.setattr(auto_nav.Node, 'create_client', lambda self, srv, name:
                        inference if name == 'set_inference_enabled'
                        else services.setdefault(name, Mock(call_async=Mock(side_effect=lambda req: Future()))))
    monkeypatch.setattr(auto_nav, 'ActionClient',
                        lambda self, action, name: clients.setdefault(name, ActionClient()))
    node = auto_nav.AutoNav()
    node.inference_control.set_enabled(True)
    node.is_running = True
    node.waypoints = [(1.0, 2.0), (3.0, 4.0), (0.0, 0.0)]
    node.home_x = node.home_y = 0.0
    for service in services.values():
        service.reset_mock()
    return SimpleNamespace(node=node, clock=clock, nav=clients['navigate_to_pose'],
                           tracking=clients['recycle_tracking_action'],
                           recycle=clients['recycle_action'], inference=inference,
                           servo=services['control_servo'], pantilt=services['control_pantilt'],
                           publishers=publishers)


@pytest.mark.parametrize('terminal_status', [
    auto_nav.GoalStatus.STATUS_CANCELED,
    auto_nav.GoalStatus.STATUS_SUCCEEDED,
    auto_nav.GoalStatus.STATUS_ABORTED,
])
def test_detection_before_navigation_acceptance_starts_one_pickup(nav_rig, terminal_status):
    """수락 전 검출을 보관하고 이동 종료 상태와 관계없이 수거를 한 번 연결한다."""
    rig = nav_rig
    rig.node.send_next_goal()
    rig.node.object_callback(detection(x=250.0))
    rig.node.object_callback(detection(x=320.0))
    assert not rig.node.object_found
    assert rig.node.previous_object_id is None
    rig.servo.call_async.assert_not_called()
    assert not rig.tracking.requests

    handle = accept(rig.nav)
    assert rig.node.object_found
    handle.cancel_goal_async.assert_called_once()
    assert rig.servo.call_async.call_count == 1
    assert not rig.tracking.requests
    rig.node.object_callback(detection(x=330.0))
    handle.cancel_goal_async.assert_called_once()
    assert rig.servo.call_async.call_count == 1
    handle.finish(terminal_status)
    assert len(rig.tracking.requests) == 1
    assert rig.tracking.requests[0][0].target_x == 330.0
    assert len(rig.nav.requests) == 1


def test_failure_can_retry_immediately_with_detection_during_patrol_acceptance(nav_rig):
    """실패와 같은 시각의 재검출도 버리지 않고 다음 순찰 수락 직후 다시 수거한다."""
    rig = nav_rig
    rig.node.send_next_goal()
    first_nav = accept(rig.nav)
    rig.node.object_callback(detection())
    first_nav.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    first_pickup = accept(rig.tracking)
    first_pickup.finish(auto_nav.GoalStatus.STATUS_ABORTED)
    assert len(rig.nav.requests) == 2
    assert not rig.node.object_found

    rig.node.object_callback(detection(x=310.0))
    assert not rig.node.object_found
    resumed_nav = accept(rig.nav)
    resumed_nav.cancel_goal_async.assert_called_once()
    resumed_nav.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    assert len(rig.tracking.requests) == 2
    assert rig.tracking.requests[-1][0].target_x == 310.0
    assert rig.clock.now == 0.0


@pytest.mark.parametrize('age,expect_pickup', [(2.0, True), (2.01, False)])
def test_pending_detection_has_maximum_age_not_a_minimum_wait(nav_rig, age, expect_pickup):
    """보관 시간 상한까지만 검출을 사용하며 만료 뒤 새 검출은 바로 처리한다."""
    rig = nav_rig
    rig.node.send_next_goal()
    rig.node.object_callback(detection())
    rig.clock.now = age
    handle = accept(rig.nav)
    assert rig.node.object_found == expect_pickup
    assert handle.cancel_goal_async.call_count == int(expect_pickup)
    if not expect_pickup:
        rig.servo.call_async.assert_not_called()
        rig.node.object_callback(detection())
        handle.cancel_goal_async.assert_called_once()
        assert rig.node.object_found


@pytest.mark.parametrize('invalid', [
    detection(class_id=-1), detection(x=float('nan')),
    SimpleNamespace(id=0, coord=[340.0, 300.0, 0.0, 80.0]),
])
def test_new_invalid_observation_clears_queued_detection(nav_rig, invalid):
    """수락 전에 대상이 사라지거나 좌표가 무효가 되면 이전 검출로 수거하지 않는다."""
    rig = nav_rig
    rig.node.send_next_goal()
    rig.node.object_callback(detection())
    rig.node.object_callback(invalid)
    handle = accept(rig.nav)
    handle.cancel_goal_async.assert_not_called()
    rig.servo.call_async.assert_not_called()
    assert not rig.node.object_found


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
@pytest.mark.parametrize('terminal_status', [
    auto_nav.GoalStatus.STATUS_CANCELED,
    auto_nav.GoalStatus.STATUS_SUCCEEDED,
    auto_nav.GoalStatus.STATUS_ABORTED,
])
def test_stop_during_navigation_acceptance_wins_over_pending_detection(
        nav_rig, command, terminal_status):
    """STOP·배터리 부족은 보관된 검출보다 우선하고 이동 종료 후 HOME으로 복귀한다."""
    rig = nav_rig
    rig.node.send_next_goal()
    rig.node.object_callback(detection())
    rig.node.command_callback(SimpleNamespace(data=command))
    rig.node.object_callback(detection())
    handle = accept(rig.nav)
    handle.cancel_goal_async.assert_called_once()
    rig.servo.call_async.assert_not_called()
    handle.finish(terminal_status)
    assert not rig.tracking.requests
    assert rig.node.is_returning_home
    assert len(rig.nav.requests) == 2
    home = rig.nav.requests[-1][0].pose.pose.position
    assert (home.x, home.y) == (0.0, 0.0)


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
def test_stop_after_pickup_cancellation_started_prevents_pickup(nav_rig, command):
    """물체 때문에 이동 취소를 요청한 뒤 STOP이 오면 서보를 닫고 HOME으로 간다."""
    rig = nav_rig
    rig.node.send_next_goal()
    handle = accept(rig.nav)
    rig.node.object_callback(detection())
    rig.node.command_callback(SimpleNamespace(data=command))
    rig.node.object_callback(detection())
    handle.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)
    assert not rig.tracking.requests
    assert rig.node.is_returning_home
    last_servo = rig.servo.call_async.call_args.args[0]
    assert (last_servo.angle1, last_servo.angle2) == (0.0, 0.0)


def test_rejected_navigation_discards_pending_detection_and_allows_next_goal(nav_rig):
    """이동 거절 뒤 발견 상태가 남지 않고 다음 목표의 새 검출로 수거할 수 있다."""
    rig = nav_rig
    rig.node.send_next_goal()
    rig.node.object_callback(detection())
    rig.nav.requests[-1][1].set_result(SimpleNamespace(accepted=False))
    assert len(rig.nav.requests) == 2
    assert not rig.node.object_found
    rig.servo.call_async.assert_not_called()
    handle = accept(rig.nav)
    handle.cancel_goal_async.assert_not_called()
    rig.node.object_callback(detection())
    handle.cancel_goal_async.assert_called_once()
    handle.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    assert len(rig.tracking.requests) == 1


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
def test_stop_then_rejected_navigation_returns_home_without_pickup(nav_rig, command):
    """STOP 대기 중 이동 요청이 거절돼도 순찰 재시도나 수거 대신 HOME으로 간다."""
    rig = nav_rig
    rig.node.send_next_goal()
    rig.node.object_callback(detection())
    rig.node.command_callback(SimpleNamespace(data=command))
    rig.nav.requests[-1][1].set_result(SimpleNamespace(accepted=False))
    assert rig.node.is_returning_home
    assert not rig.tracking.requests
    rig.servo.call_async.assert_not_called()
    assert len(rig.nav.requests) == 2


def test_detections_outside_patrol_do_not_leave_a_found_state(nav_rig):
    """이동 요청 전이나 HOME 복귀 중 관측으로 서보를 열거나 수거를 시작하지 않는다."""
    rig = nav_rig
    rig.node.object_callback(detection())
    assert not rig.node.object_found
    rig.node.is_returning_home = True
    rig.node.send_goal(0.0, 0.0)
    rig.node.object_callback(detection())
    handle = accept(rig.nav)
    rig.node.object_callback(detection())
    handle.cancel_goal_async.assert_not_called()
    rig.servo.call_async.assert_not_called()
    assert not rig.tracking.requests


@pytest.mark.parametrize('value', [0.0, -1.0, float('inf'), float('nan')])
def test_invalid_pending_detection_age_is_rejected(nav_rig, monkeypatch, value):
    """유효하지 않은 보관 시간 설정은 주행 시작 전에 거절한다."""
    monkeypatch.setattr(auto_nav.Node, 'declare_parameter',
                        lambda *args: SimpleNamespace(value=value))
    with pytest.raises(ValueError, match='pending_detection_max_age_sec'):
        auto_nav.AutoNav()


def start_unload(rig):
    """캔을 적재한 상태로 하역을 요청한다."""
    rig.node.collected_count = 2
    rig.node.previous_object_id = 0
    rig.node.launch_recycle_action()
    return accept(rig.recycle)


def selected_class(rig):
    return rig.publishers['/selected_recycle_class'].publish.call_args.args[0].data


def test_selected_class_is_not_adopted_until_patrol_handoff(nav_rig):
    rig = nav_rig
    assert selected_class(rig) == -1
    rig.node.send_next_goal()
    rig.node.object_callback(detection(class_id=3))
    assert selected_class(rig) == -1
    accept(rig.nav)
    assert selected_class(rig) == 3


def test_paper_prediction_during_trash_pickup_keeps_display_and_unload_class(nav_rig):
    rig = nav_rig
    rig.node.send_next_goal()
    accept(rig.nav)
    rig.node.object_callback(detection(class_id=3))
    assert selected_class(rig) == 3
    rig.node.object_callback(detection(class_id=1))
    assert rig.node.object_id == 1
    assert rig.node.previous_object_id == 3
    assert selected_class(rig) == 3
    rig.node.launch_recycle_action()
    assert rig.recycle.requests[-1][0].index == selected_class(rig) == 3


def test_failed_first_pickup_clears_selection_and_retry_can_adopt_paper(nav_rig):
    rig = nav_rig
    rig.node.send_next_goal()
    patrol = accept(rig.nav)
    rig.node.object_callback(detection(class_id=3))
    patrol.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    pickup = accept(rig.tracking)
    pickup.finish(auto_nav.GoalStatus.STATUS_ABORTED)
    assert selected_class(rig) == -1
    accept(rig.nav)
    rig.node.object_callback(detection(class_id=1))
    assert selected_class(rig) == 1


@pytest.mark.parametrize('collected', [0, 1])
def test_failed_pickup_keeps_selection_only_when_basket_has_items(nav_rig, collected):
    rig = nav_rig
    rig.node.send_next_goal()
    accept(rig.nav)
    rig.node.object_callback(detection(class_id=3))
    rig.node.collected_count = collected
    rig.node.resume_after_tracking_failure('test')
    assert selected_class(rig) == (3 if collected else -1)


@pytest.mark.parametrize('collected', [0, 1])
def test_stop_keeps_selection_only_for_items_already_collected(nav_rig, collected):
    rig = nav_rig
    rig.node.send_next_goal()
    patrol = accept(rig.nav)
    rig.node.object_callback(detection(class_id=3))
    rig.node.collected_count = collected
    rig.node.command_callback(SimpleNamespace(data='STOP'))
    patrol.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    assert selected_class(rig) == (3 if collected else -1)


def test_unload_completion_clears_selected_class_before_patrol_resumes(nav_rig):
    rig = nav_rig
    rig.node.send_next_goal()
    accept(rig.nav)
    rig.node.object_callback(detection(class_id=3))
    rig.node.collected_count = 1
    rig.node.launch_recycle_action()
    unload = accept(rig.recycle)
    assert selected_class(rig) == 3
    unload.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert selected_class(rig) == -1


def test_basket_check_keeps_inference_until_unload_decision(nav_rig):
    """수거 성공 직후에는 계속 추론하고 적재 상태 확인이 끝난 뒤에만 끈다."""
    rig = nav_rig
    rig.node.object_found = True
    rig.node.previous_object_id = 0
    result = Future()
    result.set_result(SimpleNamespace(status=auto_nav.GoalStatus.STATUS_SUCCEEDED,
                                     result=SimpleNamespace(success=True)))
    rig.node.recycle_tracking_result_callback(result)
    assert all(enabled for enabled, _ in rig.inference.requests)
    observation = detection()
    observation.min_y = 100.0
    rig.node.object_callback(observation)
    rig.node.check_recycle_condition_callback()
    assert not rig.inference.requests[-1][0]
    assert len(rig.recycle.requests) == 1
    assert rig.node.y_min is None


def test_nonfull_basket_returns_to_patrol_without_pausing_inference(nav_rig):
    rig = nav_rig
    rig.node.object_found = True
    rig.node.check_timer = Mock()
    rig.node.y_min = 300.0
    rig.node.check_recycle_condition_callback()
    assert not rig.recycle.requests
    assert all(enabled for enabled, _ in rig.inference.requests)


def full_patrol(rig):
    rig.node.waypoints = [(float(index), 1.0) for index in range(6)] + [(0.0, 0.0)]
    rig.node.current_idx = 4
    rig.node.send_next_goal()
    return accept(rig.nav)


def test_detection_on_the_way_to_point_four_still_starts_pickup(nav_rig):
    patrol = full_patrol(nav_rig)
    nav_rig.node.object_callback(detection())
    patrol.cancel_goal_async.assert_called_once()
    patrol.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    assert len(nav_rig.tracking.requests) == 1
    assert nav_rig.node.inference_control.enabled


@pytest.mark.parametrize('service_state', ['ready', 'delayed', 'unavailable'])
def test_point_four_to_five_and_home_ignores_detection_without_waiting_for_off(
        nav_rig, monkeypatch, service_state):
    """4번 도착 직후 차단하고, 중지 응답과 무관하게 5번을 거쳐 HOME까지 이동한다."""
    rig = nav_rig
    shutdown = Mock()
    monkeypatch.setattr(auto_nav.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(auto_nav.rclpy, 'shutdown', shutdown)
    patrol = full_patrol(rig)
    assert rig.node.inference_control.ready
    rig.inference.auto_respond = service_state != 'delayed'
    rig.inference.available = service_state != 'unavailable'
    patrol.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)

    for expected in ((5.0, 1.0), (0.0, 0.0)):
        shutdown.assert_not_called()
        assert not rig.node.inference_control.enabled
        position = rig.nav.requests[-1][0].pose.pose.position
        assert (position.x, position.y) == expected
        rig.node.object_callback(detection())  # 이동 요청 수락 전의 늦은 검출
        assert rig.node._pending_detection is None
        returning = accept(rig.nav)
        rig.node.object_callback(detection())  # 이동 중의 늦은 검출
        assert rig.node.target_x is None
        returning.cancel_goal_async.assert_not_called()
        returning.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)

    shutdown.assert_called_once()
    assert len(rig.nav.requests) == 3
    assert not rig.tracking.requests and not rig.recycle.requests
    rig.servo.call_async.assert_not_called()


@pytest.mark.parametrize('status', [auto_nav.GoalStatus.STATUS_ABORTED,
                                  auto_nav.GoalStatus.STATUS_CANCELED])
def test_return_waypoint_retry_keeps_inference_paused(nav_rig, status):
    rig = nav_rig
    full_patrol(rig).finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)
    accept(rig.nav).finish(status)
    assert not rig.node.inference_control.enabled
    assert rig.node.current_idx == 5
    returning = accept(rig.nav)
    rig.node.object_callback(detection())
    returning.cancel_goal_async.assert_not_called()
    assert not rig.tracking.requests


def test_short_selected_path_keeps_patrol_detection_and_pauses_for_home(nav_rig):
    rig = nav_rig
    rig.node.current_idx = 1
    rig.node.send_next_goal()
    assert rig.node.inference_control.ready
    accept(rig.nav).finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)
    assert rig.node.current_idx == 2  # 짧은 선택 경로에서도 마지막 HOME은 중지 대상
    assert not rig.node.inference_control.enabled
    returning = accept(rig.nav)
    rig.node.object_callback(detection())
    returning.cancel_goal_async.assert_not_called()


def test_home_arrival_with_load_unloads_and_reenables_patrol(nav_rig):
    rig = nav_rig
    rig.node.collected_count, rig.node.previous_object_id = 1, 3
    full_patrol(rig).finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)
    accept(rig.nav).finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)  # 5번
    accept(rig.nav).finish(auto_nav.GoalStatus.STATUS_SUCCEEDED)  # HOME
    assert rig.recycle.requests[-1][0].index == 3
    assert not rig.node.inference_control.enabled
    accept(rig.recycle).finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert rig.node.current_idx == 0
    assert rig.node.inference_control.ready
    assert not rig.node.object_found
    patrol = accept(rig.nav)
    rig.node.object_callback(detection(class_id=1))
    patrol.cancel_goal_async.assert_called_once()


def test_unload_waits_for_on_ack_before_sending_patrol(nav_rig):
    rig = nav_rig
    handle = start_unload(rig)
    assert not rig.node.inference_control.enabled
    rig.node.object_callback(detection())
    assert rig.node.target_x is None
    rig.inference.auto_respond = False
    handle.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert rig.inference.requests[-1][0]
    assert not rig.nav.requests
    assert rig.node.object_found
    rig.node.object_callback(detection())
    assert rig.node._pending_detection is None
    rig.inference.requests[-1][1].set_result(SimpleNamespace(success=True))
    assert len(rig.nav.requests) == 1
    assert not rig.node.object_found
    assert rig.node.collected_count == 0
    assert rig.node.previous_object_id is None
    handle = accept(rig.nav)
    rig.node.object_callback(detection(class_id=2))
    handle.cancel_goal_async.assert_called_once()


def test_late_off_ack_cannot_resume_patrol(nav_rig):
    rig = nav_rig
    rig.inference.auto_respond = False
    handle = start_unload(rig)
    late_off = rig.inference.requests[-1][1]
    handle.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert not rig.nav.requests
    assert not rig.inference.requests[-1][0]
    late_off.set_result(SimpleNamespace(success=True))
    assert rig.inference.requests[-1][0]
    assert not rig.nav.requests
    rig.inference.requests[-1][1].set_result(SimpleNamespace(success=True))
    assert len(rig.nav.requests) == 1


@pytest.mark.parametrize('phase', ['goal_pending', 'unloading', 'resume_pending'])
@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
def test_stop_during_unload_or_resume_never_restarts_patrol(nav_rig, phase, command):
    rig = nav_rig
    if phase == 'goal_pending':
        rig.node.launch_recycle_action()
        rig.node.command_callback(SimpleNamespace(data=command))
        handle = accept(rig.recycle)
        handle.cancel_goal_async.assert_called_once()
        handle.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    else:
        handle = start_unload(rig)
        if phase == 'resume_pending':
            rig.inference.auto_respond = False
            handle.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
            rig.node.command_callback(SimpleNamespace(data=command))
            rig.inference.requests[-1][1].set_result(SimpleNamespace(success=True))
        else:
            rig.node.command_callback(SimpleNamespace(data=command))
            handle.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    assert rig.node.is_returning_home
    assert not rig.node.inference_control.enabled
    assert len(rig.nav.requests) == 1
    position = rig.nav.requests[0][0].pose.pose.position
    assert (position.x, position.y) == (0.0, 0.0)


@pytest.mark.parametrize('failure', ['rejected', 'send_exception', 'goal_exception',
                                     'result_exception', 'aborted'])
def test_unload_errors_release_inference_pause(nav_rig, monkeypatch, failure):
    rig = nav_rig
    monkeypatch.setattr(auto_nav.rclpy, 'ok', lambda: False)
    if failure == 'send_exception':
        rig.node._recycle_client.send_goal_async = Mock(side_effect=RuntimeError('send failed'))
    rig.node.launch_recycle_action()
    if failure == 'rejected':
        rig.recycle.requests[-1][1].set_result(SimpleNamespace(accepted=False))
    elif failure == 'goal_exception':
        rig.recycle.requests[-1][1].set_exception(RuntimeError('goal lost'))
    elif failure in ('result_exception', 'aborted'):
        handle = accept(rig.recycle)
        if failure == 'result_exception':
            handle.result.set_exception(RuntimeError('result lost'))
        else:
            handle.finish(auto_nav.GoalStatus.STATUS_ABORTED)
    assert rig.node.inference_control.enabled
    assert rig.inference.requests[-1][0]
    assert not rig.nav.requests


def test_startup_waits_for_detector_and_stop_wins_while_waiting(nav_rig):
    rig = nav_rig
    rig.inference.auto_respond = False
    rig.node.inference_control.confirmed = None
    rig.node.is_running = False
    path = SimpleNamespace(poses=[SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=x, y=y))) for x, y in rig.node.waypoints])
    rig.node.path_callback(path)
    assert not rig.nav.requests
    rig.node.command_callback(SimpleNamespace(data='STOP'))
    rig.inference.requests[-1][1].set_result(SimpleNamespace(success=True))
    assert len(rig.nav.requests) == 1
    assert rig.node.is_returning_home


def start_basket_check(rig, collected=0):
    """TRASH를 임시 채택한 뒤 실제 수거 결과 콜백으로 3초 확인을 시작한다."""
    rig.node.object_found = True
    rig.node.collected_count = collected
    rig.node.previous_object_id = 3
    rig.node.publish_selected_class(3)
    result = Future()
    result.set_result(SimpleNamespace(status=auto_nav.GoalStatus.STATUS_SUCCEEDED,
                                     result=SimpleNamespace(success=True)))
    rig.node.recycle_tracking_result_callback(result)
    return rig.node.pantilt_future


def basket_observation(rig, name='paper', stamp_sec=None, x=320.0, camera_stamp_sec=None):
    """PC 수신 시각과 별도 카메라 촬영 시각을 가진 관측을 만든다."""
    message = Detection2D()
    batch = Detection2DArray()
    stamp = int((100 + rig.clock.now if stamp_sec is None else stamp_sec) * 1e9)
    batch.header.stamp = Time(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)
    camera_stamp = stamp if camera_stamp_sec is None else int(camera_stamp_sec * 1e9)
    message.header.stamp = Time(sec=camera_stamp // 1_000_000_000, nanosec=camera_stamp % 1_000_000_000)
    message.bbox.center.position.x, message.bbox.center.position.y = x, 300.0
    message.bbox.size_x, message.bbox.size_y = 60.0, 80.0
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id, result.hypothesis.score = name, 0.9
    message.results = [result]
    batch.detections = [message]
    return batch


def feed_basket(rig, names):
    for name in names:
        rig.clock.now += 0.1
        rig.node.class_observation_callback(basket_observation(rig, name))


def start_approach(rig, class_id=3):
    rig.node.send_next_goal()
    patrol = accept(rig.nav)
    rig.node.object_callback(detection(class_id=class_id))
    patrol.finish(auto_nav.GoalStatus.STATUS_CANCELED)
    return accept(rig.tracking)


def feed_approach(rig, name, score, count):
    for _ in range(count):
        rig.clock.now += 0.01
        batch = basket_observation(rig, name)
        batch.detections[0].results[0].hypothesis.score = score
        rig.node.class_observation_callback(batch)


def test_approach_mean_beats_majority_and_basket_cannot_overwrite_it(nav_rig):
    rig = nav_rig
    logger = Mock()
    rig.node.get_logger = lambda: logger
    pickup = start_approach(rig)
    feed_approach(rig, 'trash', 0.630, 92)
    feed_approach(rig, 'paper', 0.802, 47)
    feed_approach(rig, 'can', 0.99, 1)  # 1회의 높은 점수는 후보가 될 수 없다.
    assert selected_class(rig) == 3  # 10회를 채워도 수거 성공 전에는 임시 종류 유지
    pickup.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert selected_class(rig) == rig.node.previous_object_id == 1
    assert rig.node._basket_deadline is None
    rig.node.pantilt_future.set_result(SimpleNamespace(success=True))
    feed_basket(rig, ['trash'] * 11)
    rig.clock.now += 3.0
    # 기존 적재량 판정은 계속 받아 하역을 연결해야 한다.
    observation = detection(class_id=1)
    observation.min_y = 100.0
    rig.node.object_callback(observation)
    rig.node.check_recycle_condition_callback()
    assert rig.recycle.requests[-1][0].index == selected_class(rig) == 1
    assert any('접근 분류 확정' in call.args[0] and 'paper=47/140회' in call.args[0]
               and '평균 차이 0.172' in call.args[0] for call in logger.info.call_args_list)


@pytest.mark.parametrize('paper_count,paper_score,trash_count,trash_score,confirmed', [
    (10, 0.637, 10, 0.668, False),  # 충분히 관측했어도 평균 차이가 작다.
    (10, 0.80, 10, 0.80, False),
    (9, 0.90, 9, 0.60, False),
    (10, 0.85, 10, 0.80, True),    # 차이 0.05 경계값
    (10, 0.80, 9, 0.95, True),     # 최소 횟수를 충족한 클래스 하나
])
def test_approach_thresholds_choose_confirmation_or_basket(
        nav_rig, paper_count, paper_score, trash_count, trash_score, confirmed):
    rig = nav_rig
    pickup = start_approach(rig, class_id=1)
    feed_approach(rig, 'paper', paper_score, paper_count)
    feed_approach(rig, 'trash', trash_score, trash_count)
    pickup.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert (rig.node._basket_deadline is None) == confirmed
    rig.node.pantilt_future.set_result(SimpleNamespace(success=True))
    rig.clock.now += 0.5
    feed_basket(rig, ['trash'] * 3)
    rig.clock.now += 3.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == (1 if confirmed else 3)


def test_approach_only_counts_fresh_unique_valid_frames_for_current_pickup(nav_rig):
    rig = nav_rig
    feed_approach(rig, 'paper', 0.99, 10)  # 발견 채택 전은 제외
    start_approach(rig)
    for kind in ('old', 'future', 'empty', 'person', 'nan', 'bad_box'):
        rig.clock.now += 0.01
        batch = basket_observation(rig)
        if kind in ('old', 'future'):
            batch.header.stamp.sec = 0 if kind == 'old' else 10000
        elif kind == 'empty':
            batch.detections = []
        elif kind == 'person':
            batch.detections[0].results[0].hypothesis.class_id = 'person'
        elif kind == 'nan':
            batch.detections[0].results[0].hypothesis.score = float('nan')
        else:
            batch.detections[0].bbox.size_x = 0.0
        rig.node.class_observation_callback(batch)
    assert not rig.node._approach_counts
    # Pi 시각이 PC와 달라도 허용하되 같은 촬영 시각은 두 번 세지 않는다.
    for camera_stamp in (500.0, 500.0, 499.0, 501.0, 0.0):
        rig.clock.now += 0.01
        batch = basket_observation(rig, camera_stamp_sec=camera_stamp)
        rig.node.class_observation_callback(batch)
        rig.node.class_observation_callback(batch)
    assert dict(rig.node._approach_counts) == {1: 3}


@pytest.mark.parametrize('ending', ['failure', 'STOP', 'BATTERY_LOW'])
def test_approach_statistics_are_discarded_on_failure_or_stop(nav_rig, ending):
    rig = nav_rig
    pickup = start_approach(rig)
    feed_approach(rig, 'paper', 0.95, 10)
    if ending == 'failure':
        pickup.finish(auto_nav.GoalStatus.STATUS_ABORTED)
    else:
        rig.node.command_callback(SimpleNamespace(data=ending))
        pickup.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    feed_approach(rig, 'paper', 0.95, 10)
    assert not rig.node._approach_counts
    assert rig.node._approach_after_ns is None
    assert selected_class(rig) == -1
    assert not rig.recycle.requests
    if ending == 'failure':
        resumed = accept(rig.nav)
        rig.node.object_callback(detection())
        resumed.finish(auto_nav.GoalStatus.STATUS_CANCELED)
        retry = accept(rig.tracking)
        retry.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
        assert rig.node._basket_deadline is not None  # 이전 10회로 확정하지 않는다.


def test_later_pickup_does_not_change_existing_load_using_approach_means(nav_rig):
    rig = nav_rig
    rig.node.collected_count, rig.node.previous_object_id = 1, 3
    pickup = start_approach(rig)
    feed_approach(rig, 'paper', 0.99, 10)
    pickup.finish(auto_nav.GoalStatus.STATUS_SUCCEEDED, success=True)
    assert selected_class(rig) == 3
    assert not rig.node._approach_counts
    assert rig.node._basket_deadline is None


@pytest.mark.parametrize('name,value', [
    ('approach_min_samples', 0), ('approach_min_samples', 1.5),
    ('approach_mean_margin', 0.0), ('approach_mean_margin', 1.01),
    ('approach_mean_margin', float('nan')), ('approach_mean_margin', float('inf')),
])
def test_invalid_approach_settings_are_rejected(nav_rig, monkeypatch, name, value):
    monkeypatch.setattr(auto_nav.Node, 'declare_parameter', lambda self, key, default:
                        SimpleNamespace(value=value if key == name else default))
    with pytest.raises(ValueError, match='접근 분류'):
        auto_nav.AutoNav()


@pytest.mark.parametrize('names,expected', [
    (['paper'] * 3, 1),
    (['paper', 'trash', 'paper', 'paper', 'paper'], 1),
    (['paper', 'trash'] * 3, 3),
    (['paper', 'paper', 'trash'], 3),
    (['paper'] * 2, 3),
    ([], 3),
])
def test_first_basket_consensus_controls_display_and_unload(nav_rig, names, expected):
    rig = nav_rig
    tilt = start_basket_check(rig)
    rig.clock.now = 0.4
    tilt.set_result(SimpleNamespace(success=True))
    rig.clock.now = 0.9
    feed_basket(rig, names)
    assert selected_class(rig) == 3  # 확인 도중의 다수결로 미리 변경하지 않는다.
    rig.clock.now = 3.0
    rig.node.y_min = 100.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == rig.node.previous_object_id == expected
    assert rig.recycle.requests[-1][0].index == expected
    feed_basket(rig, ['plastic'] * 5)
    assert selected_class(rig) == expected


def test_basket_uses_fresh_unique_images_after_successful_tilt(nav_rig):
    rig = nav_rig
    tilt = start_basket_check(rig)
    rig.clock.now = 0.2
    rig.node.basket_class_callback(basket_observation(rig))  # 응답 전
    rig.clock.now = 0.4
    tilt.set_result(SimpleNamespace(success=True))
    rig.clock.now = 0.6
    rig.node.basket_class_callback(basket_observation(rig))  # 안정화 중
    rig.clock.now = 1.0
    for stamp in (0.0, 100.2, 100.6, 102.0):
        rig.node.basket_class_callback(basket_observation(rig, stamp_sec=stamp))
    valid = basket_observation(rig)
    for _ in range(5):
        rig.node.basket_class_callback(valid)  # 중복 발행은 1회만 계산
    rig.node.basket_class_callback(basket_observation(rig, stamp_sec=100.9))
    assert dict(rig.node._basket_votes) == {1: 1}
    rig.clock.now = 3.0
    feed_basket(rig, ['paper'] * 3)  # 타이머 콜백이 늦어져도 3초 밖은 제외
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 3


@pytest.mark.parametrize('failure', ['pending', 'unsuccessful', 'exception', 'late'])
def test_tilt_failure_or_late_reply_keeps_provisional_class(nav_rig, failure):
    rig = nav_rig
    tilt = start_basket_check(rig)
    if failure == 'unsuccessful':
        tilt.set_result(SimpleNamespace(success=False))
    elif failure == 'exception':
        tilt.set_exception(RuntimeError('connection lost'))
    rig.clock.now = 1.0
    feed_basket(rig, ['paper'] * 5)
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    if failure == 'late':
        tilt.set_result(SimpleNamespace(success=True))
    assert rig.node._basket_after_ns is None
    assert selected_class(rig) == 3


def test_late_tilt_reply_from_previous_pickup_cannot_arm_new_check(nav_rig):
    rig = nav_rig
    old_tilt = start_basket_check(rig)
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    start_basket_check(rig)
    old_tilt.set_result(SimpleNamespace(success=True))
    feed_basket(rig, ['paper'] * 5)
    assert not rig.node._basket_votes
    assert rig.node._basket_after_ns is None


@pytest.mark.parametrize('collected', [1, 2])
def test_later_items_do_not_reclassify_existing_load(nav_rig, collected):
    rig = nav_rig
    tilt = start_basket_check(rig, collected=collected)
    tilt.set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    feed_basket(rig, ['paper'] * 5)
    rig.clock.now = 3.0
    rig.node.y_min = 100.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == rig.recycle.requests[-1][0].index == 3
    assert not rig.node._basket_votes


@pytest.mark.parametrize('command', ['STOP', 'BATTERY_LOW'])
def test_stop_during_basket_check_discards_votes_and_goes_home(nav_rig, command):
    rig = nav_rig
    tilt = start_basket_check(rig)
    tilt.set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    feed_basket(rig, ['paper'] * 3)
    rig.node.command_callback(SimpleNamespace(data=command))
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 3
    assert rig.node.is_returning_home
    assert not rig.recycle.requests
    assert len(rig.nav.requests) == 1


def test_different_position_and_nonrecyclable_observations_do_not_vote(nav_rig):
    rig = nav_rig
    start_basket_check(rig).set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    feed_basket(rig, ['paper'] * 2)
    for name, x in [('paper', 500.0), ('person', 320.0), ('unknown', 320.0)]:
        rig.clock.now += 0.1
        rig.node.basket_class_callback(basket_observation(rig, name, x=x))
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 3


def test_corrected_class_filters_next_pickup(nav_rig):
    rig = nav_rig
    start_basket_check(rig).set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    feed_basket(rig, ['paper'] * 3)
    rig.clock.now = 3.0
    rig.node.y_min = 300.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 1
    rig.node.object_found = False
    rig.node.send_next_goal()
    patrol = accept(rig.nav)
    rig.node.object_callback(detection(class_id=3))
    patrol.cancel_goal_async.assert_not_called()
    rig.node.object_callback(detection(class_id=1))
    patrol.cancel_goal_async.assert_called_once()


@pytest.mark.parametrize('name,value', [
    ('basket_min_samples', 2),
    ('basket_agreement_ratio', 0.5),
    ('basket_agreement_ratio', 1.01),
    ('basket_agreement_ratio', float('nan')),
    ('basket_settle_sec', -0.1),
    ('basket_settle_sec', 3.0),
    ('basket_settle_sec', float('inf')),
    ('basket_settle_sec', float('nan')),
])
def test_invalid_basket_settings_are_rejected(nav_rig, monkeypatch, name, value):
    monkeypatch.setattr(auto_nav.Node, 'declare_parameter', lambda self, key, default:
                        SimpleNamespace(value=value if key == name else default))
    with pytest.raises(ValueError, match='수거함 재분류'):
        auto_nav.AutoNav()


def test_custom_basket_settings_control_settling_and_consensus(nav_rig):
    rig = nav_rig
    rig.node.basket_min_samples = 5
    rig.node.basket_agreement_ratio = 1.0
    rig.node.basket_settle_sec = 1.0
    tilt = start_basket_check(rig)
    rig.clock.now = 0.4
    tilt.set_result(SimpleNamespace(success=True))
    rig.clock.now = 0.9
    feed_basket(rig, ['paper'] * 3)
    assert not rig.node._basket_votes
    rig.clock.now = 1.5
    feed_basket(rig, ['paper', 'paper', 'trash', 'paper', 'paper'])
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 3  # 엄격한 100% 일치 설정을 적용한다.


@pytest.mark.parametrize('invalid', ['no_result', 'nan_center', 'zero_size', 'nan_score'])
def test_invalid_basket_observations_do_not_complete_consensus(nav_rig, invalid):
    rig = nav_rig
    start_basket_check(rig).set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    feed_basket(rig, ['paper'] * 2)
    rig.clock.now += 0.1
    observation = basket_observation(rig)
    detection = observation.detections[0]
    if invalid == 'no_result':
        detection.results = []
    elif invalid == 'nan_center':
        detection.bbox.center.position.x = float('nan')
    elif invalid == 'zero_size':
        detection.bbox.size_x = 0.0
    else:
        detection.results[0].hypothesis.score = float('nan')
    rig.node.basket_class_callback(observation)
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 3


@pytest.mark.parametrize('camera_offset', [-100.0, 2.0, 800.0, None])
def test_basket_reclassifies_despite_camera_clock_offset_or_zero_stamp(nav_rig, camera_offset):
    rig = nav_rig
    start_basket_check(rig).set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    for _ in range(4):
        rig.clock.now += 0.1
        camera_time = 0.0 if camera_offset is None else 100 + rig.clock.now + camera_offset
        rig.node.basket_class_callback(basket_observation(rig, camera_stamp_sec=camera_time))
    rig.clock.now = 3.0
    rig.node.y_min = 100.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == rig.recycle.requests[-1][0].index == 1


def test_duplicate_camera_frame_with_new_pc_receipt_time_cannot_add_votes(nav_rig):
    rig = nav_rig
    start_basket_check(rig).set_result(SimpleNamespace(success=True))
    rig.clock.now = 1.0
    for camera_time in (500.0, 500.0, 499.9, 500.0, 500.1):
        rig.clock.now += 0.1
        rig.node.basket_class_callback(basket_observation(rig, camera_stamp_sec=camera_time))
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    assert selected_class(rig) == 3
    assert rig.node._basket_skips['중복/역순'] == 3


@pytest.mark.parametrize('case,reason', [
    ('no_tilt', '틸트 성공 응답 없음'),
    ('no_observations', '재분류 관측 수신 없음'),
    ('old_frame', '안정화 전/PC 시각'),
])
def test_empty_basket_vote_reports_why_samples_were_not_used(nav_rig, case, reason):
    rig = nav_rig
    logger = Mock()
    rig.node.get_logger = lambda: logger
    tilt = start_basket_check(rig)
    if case != 'no_tilt':
        tilt.set_result(SimpleNamespace(success=True))
    if case == 'old_frame':
        rig.clock.now = 1.0
        rig.node.basket_class_callback(basket_observation(rig, stamp_sec=100.2))
    rig.clock.now = 3.0
    rig.node.check_recycle_condition_callback()
    logs = [call.args[0] for call in logger.info.call_args_list]
    assert any('수거함 분류 유지' in line and '0/0회' in line and reason in line for line in logs)
