"""가상 시간과 검출 결과로 수거의 이동 명령 및 종료 동작을 확인한다."""

from concurrent.futures import Future
from types import SimpleNamespace

from navigation import auto_nav, recycle_tracking_node as tracking

import pytest


class Clock:
    """실제 대기 없이 제어 시간과 입력 순서를 재현한다."""

    def __init__(self):
        """가상 시각과 입력 전달 함수를 초기화한다."""
        self.now = 0.0
        self.on_step = lambda now: None

    def advance(self, seconds):
        """시간 경과에 맞춰 다음 입력을 전달한다."""
        self.now = round(self.now + seconds, 6)
        self.on_step(self.now)


class Event:
    """취소 및 서비스 완료 이벤트의 대기를 가상 시간으로 처리한다."""

    def __init__(self, clock):
        """시계와 초기 이벤트 상태를 저장한다."""
        self.clock, self.flag = clock, False

    def set(self):  # noqa: A003
        """threading.Event와 같은 이름으로 완료 상태를 알린다."""
        self.flag = True

    def clear(self):
        """이벤트를 대기 상태로 되돌린다."""
        self.flag = False

    def is_set(self):
        """완료 여부를 반환한다."""
        return self.flag

    def wait(self, timeout):
        """미완료일 때만 시간을 진행한다."""
        if not self.flag:
            self.clock.advance(timeout)
        return self.flag


class Goal:
    """수거 Action의 최종 상태를 기록한다."""

    def __init__(self):
        """수거별 요청 좌표와 취소 상태를 초기화한다."""
        self.is_cancel_requested = False
        self.state = None
        self.request = tracking.RecycleActionMsg.Goal()

    def succeed(self):
        """성공 처리를 기록한다."""
        self.state = '성공'

    def abort(self):
        """실패 처리를 기록한다."""
        self.state = '실패'

    def canceled(self):
        """취소 처리를 기록한다."""
        self.state = '취소'


def detection(x=350.0, bottom=300.0, class_id=0):
    """검출기의 좌표 형식에 맞는 관측을 만든다."""
    return SimpleNamespace(id=class_id, coord=[x, bottom - 40.0, 60.0, 80.0])


@pytest.fixture
def rig(monkeypatch):
    """ROS 통신 없이 실제 노드의 제어와 서비스 완료 순서를 실행한다."""
    clock, velocities, requests, logs = Clock(), [], [], []
    client = SimpleNamespace(pending_on=False, pending_off=False, reject_off=False)

    def call_async(request):
        future = Future()
        requests.append((request.enable, future))
        if not (client.pending_on if request.enable else client.pending_off):
            future.set_result(SimpleNamespace(success=request.enable or not client.reject_off))
        return future

    client.call_async = call_async
    logger = SimpleNamespace(info=logs.append, warn=logs.append, error=logs.append)
    publisher = SimpleNamespace(publish=lambda msg: velocities.append(
        (clock.now, msg.linear.x, msg.angular.z)))
    monkeypatch.setattr(tracking, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(tracking, 'Event', lambda: Event(clock))
    monkeypatch.setattr(tracking.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(tracking.Node, '__init__', lambda self, name: None)
    monkeypatch.setattr(tracking.Node, 'declare_parameter',
                        lambda self, name, default: SimpleNamespace(value=default))
    monkeypatch.setattr(tracking.Node, 'get_logger', lambda self: logger)
    monkeypatch.setattr(tracking.Node, 'create_subscription', lambda *a, **kw: None)
    monkeypatch.setattr(tracking.Node, 'create_publisher', lambda *a, **kw: publisher)
    monkeypatch.setattr(tracking.Node, 'create_client', lambda *a, **kw: client)
    monkeypatch.setattr(tracking, 'ActionServer', lambda *a, **kw: None)
    node = tracking.RecycleTrackingNode()
    goal = Goal()

    def run():
        assert node.goal_callback(None) == tracking.GoalResponse.ACCEPT
        return node.execute_callback(goal)

    return SimpleNamespace(node=node, clock=clock, goal=goal, client=client,
                           velocities=velocities, requests=requests, logs=logs, run=run)


def test_small_error_steers_large_error_stops_and_realigns(rig):
    """이동 중 보정과 정지 재정렬을 거쳐 수거를 완료한다."""
    def feed(now):
        if now < 0.3:
            msg = detection()
        elif now < 0.6:
            msg = detection(x=330.0)
        elif now < 0.9:
            msg = detection(x=280.0)
        elif now < 1.3:
            msg = detection()
        else:
            msg = detection(bottom=440.0)
        rig.node.obj_callback(msg)

    rig.clock.on_step = feed
    result = rig.run()
    assert result.success and rig.goal.state == '성공'
    assert any(v > 0 and w > 0 for t, v, w in rig.velocities if 0.3 <= t < 0.6)
    assert all(v == 0 for t, v, w in rig.velocities if 0.6 <= t < 0.9)
    assert any(w > 0 for t, v, w in rig.velocities if 0.6 <= t < 0.9)
    assert any(v > 0 and w == 0 for t, v, w in rig.velocities if 1.0 <= t < 1.3)
    assert rig.velocities[-1][1:] == (0, 0)
    assert [enable for enable, _ in rig.requests] == [True, False]


@pytest.mark.parametrize('x,sign', [(330.0, 1), (370.0, -1)])
def test_correction_direction(rig, x, sign):
    """좌우 오차에 맞는 방향으로 이동 중 보정한다."""
    def feed(now):
        rig.node.obj_callback(detection(x=350.0 if now < 0.3 else x))
        if now >= 0.7:
            rig.goal.is_cancel_requested = True
            rig.node.cancel_callback(rig.goal)

    rig.clock.on_step = feed
    result = rig.run()
    assert any(v > 0 and w * sign > 0 for _, v, w in rig.velocities)
    assert not result.success and rig.goal.state == '취소'


@pytest.mark.parametrize('kind,expected,limit', [
    ('정렬', '정렬 시간 초과', 15.0),
    ('접근', '전체 주행 시간 초과', 40.0),
    ('재정렬 반복', '전체 주행 시간 초과', 40.0),
])
def test_fresh_detections_cannot_extend_deadline(rig, kind, expected, limit):
    """검출이 계속되거나 재정렬을 반복해도 시도 시간을 연장하지 않는다."""
    def feed(now):
        x = 250.0 if kind == '정렬' else 350.0
        if kind == '재정렬 반복' and int(now * 10) % 10 == 5:
            x = 250.0
        rig.node.obj_callback(detection(x=x))

    rig.clock.on_step = feed
    result = rig.run()
    assert not result.success and expected in result.message
    assert rig.goal.state == '실패' and rig.clock.now <= limit + 0.1
    assert rig.velocities[-1][1:] == (0, 0)
    if kind == '재정렬 반복':
        assert sum('재정렬 시작' in log for log in rig.logs) > 2


@pytest.mark.parametrize('mode', ['미검출', '수신 중단', '잘못된 좌표'])
def test_loss_without_reliable_direction_stops_before_failure(rig, mode):
    """대상이 중앙에 있었거나 검출 정보가 끊기거나 잘못되면 정지한다."""
    def feed(now):
        if now < 0.5:
            x = 330.0 if now >= 0.3 and mode != '미검출' else 350.0
            rig.node.obj_callback(detection(x=x))
        elif mode == '미검출':
            rig.node.obj_callback(detection(class_id=-1))
        elif mode == '잘못된 좌표':
            rig.node.obj_callback(detection(x=float('nan')))

    rig.clock.on_step = feed
    result = rig.run()
    assert not result.success and '재탐지 시간 초과' in result.message
    assert any(v > 0 for _, v, _ in rig.velocities)
    stop_by = 1.5 if mode == '수신 중단' else 0.5
    assert all(v == w == 0 for t, v, w in rig.velocities if t >= stop_by)
    assert rig.clock.now < 6.0


def test_one_message_cannot_complete_alignment(rig):
    """같은 검출 하나를 반복해서 읽어 정렬 성공으로 처리하지 않는다."""
    def feed(now):
        if now == 0.1:
            rig.node.obj_callback(detection(bottom=440.0))

    rig.clock.on_step = feed
    result = rig.run()
    assert not result.success
    assert not any(v or w for _, v, w in rig.velocities)
    assert not any(log.startswith('정렬 완료:') for log in rig.logs)


def test_brief_target_loss_can_resume_without_a_new_action(rig):
    """짧은 미검출 뒤 같은 수거 시도 안에서 접근을 계속한다."""
    def feed(now):
        rig.node.obj_callback(detection(
            bottom=440.0 if now >= 1.2 else 300.0,
            class_id=-1 if 0.5 <= now < 1.0 else 0))

    rig.clock.on_step = feed
    result = rig.run()
    assert result.success
    assert all(v == w == 0 for t, v, w in rig.velocities if 0.5 <= t < 1.0)
    assert any(v > 0 for t, v, w in rig.velocities if 1.0 <= t < 1.2)
    assert [enable for enable, _ in rig.requests] == [True, False]


@pytest.mark.parametrize('x,sign', [(260.0, 1), (440.0, -1)])
def test_lost_target_search_uses_latest_visible_position(rig, x, sign):
    """2초 넘게 놓친 대상도 마지막 검출 위치 방향으로 회전해 다시 찾는다."""
    rig.goal.request.target_x = 700.0 - x
    rig.goal.request.target_h = 80.0

    def feed(now):
        if now < 0.4:
            msg = detection(x=x)
        elif now < 2.9:
            msg = detection(x=0.0, class_id=-1)
        else:
            msg = detection(bottom=440.0)
        rig.node.obj_callback(msg)

    rig.clock.on_step = feed
    result = rig.run()
    searching = [(v, w) for t, v, w in rig.velocities if 0.4 <= t < 2.9]
    assert searching and all(v == 0 and w * sign > 0 for v, w in searching)
    assert result.success and rig.goal.state == '성공'
    assert rig.velocities[-1][1:] == (0, 0)
    assert [enable for enable, _ in rig.requests] == [True, False]


@pytest.mark.parametrize('x,sign', [(250.0, 1), (450.0, -1), (350.0, 0)])
def test_initial_search_uses_current_goal_without_guessing_center_direction(rig, x, sign):
    """첫 검출 전에는 이번 요청의 위치를 쓰되 중앙이면 방향을 임의로 정하지 않는다."""
    rig.goal.request.target_x = x
    rig.goal.request.target_h = 80.0
    rig.clock.on_step = lambda now: rig.node.obj_callback(detection(
        x=0.0 if now < 0.5 else 350.0, bottom=440.0,
        class_id=-1 if now < 0.5 else 0))
    result = rig.run()
    searching = [(v, w) for t, v, w in rig.velocities if 0.1 <= t < 0.5]
    assert searching and all(v == 0 for v, _ in searching)
    if sign:
        assert all(w * sign > 0 for _, w in searching)
    else:
        assert all(w == 0 for _, w in searching)
    assert result.success
    assert all(v == 0 for t, v, _ in rig.velocities if t < 0.7)


def test_reacquisition_during_approach_requires_alignment_before_forward(rig):
    """접근 중 재탐지 회전 후에는 작은 오차도 정렬을 마친 뒤 전진한다."""
    def feed(now):
        x = 330.0 if 0.3 <= now < 1.3 else 350.0
        rig.node.obj_callback(detection(
            x=x, bottom=440.0 if now >= 1.7 else 300.0,
            class_id=-1 if 0.5 <= now < 1.0 else 0))

    rig.clock.on_step = feed
    result = rig.run()
    assert any(v > 0 for t, v, _ in rig.velocities if t < 0.5)
    assert all(v == 0 and w > 0 for t, v, w in rig.velocities if 0.5 <= t < 1.3)
    assert all(v == 0 for t, v, _ in rig.velocities if 1.3 <= t < 1.5)
    assert any(v > 0 for t, v, _ in rig.velocities if 1.5 <= t < 1.7)
    assert result.success


def test_search_rotation_times_out_and_stops(rig):
    """미검출 메시지가 계속 도착해도 재탐지 회전 시간을 연장하지 않는다."""
    rig.clock.on_step = lambda now: rig.node.obj_callback(detection(
        x=450.0, class_id=0 if now < 0.4 else -1))
    result = rig.run()
    assert not result.success and '재탐지 시간 초과' in result.message
    assert any(w < 0 for t, _, w in rig.velocities if t >= 0.4)
    assert all(v == 0 for _, v, _ in rig.velocities)
    assert 4.4 <= rig.clock.now <= 4.5
    assert rig.velocities[-1][1:] == (0, 0)


@pytest.mark.parametrize('mode', ['수신 중단', '잘못된 좌표'])
def test_search_stops_when_detection_feed_becomes_unusable(rig, mode):
    """재탐지 중 수신이 끊기거나 좌표가 잘못되면 회전을 멈춘다."""
    def feed(now):
        if now < 0.4:
            rig.node.obj_callback(detection(x=250.0))
        elif now < 0.8:
            rig.node.obj_callback(detection(class_id=-1))
        elif mode == '잘못된 좌표':
            rig.node.obj_callback(detection(x=float('nan')))

    rig.clock.on_step = feed
    result = rig.run()
    assert any(w > 0 for t, _, w in rig.velocities if 0.4 <= t < 0.8)
    stop_by = 1.8 if mode == '수신 중단' else 0.8
    assert all(v == w == 0 for t, v, w in rig.velocities if t >= stop_by)
    assert not result.success and rig.clock.now <= 4.5


def test_cancel_during_search_prevents_later_motion_and_direction_reuse(rig):
    """재탐지 회전 중 STOP을 지키고 다음 수거에 이전 방향을 넘기지 않는다."""
    def feed(now):
        rig.node.obj_callback(detection(x=250.0, class_id=0 if now < 0.4 else -1))
        if now == 0.8:
            rig.goal.is_cancel_requested = True
            rig.node.cancel_callback(rig.goal)

    rig.clock.on_step = feed
    result = rig.run()
    assert any(w > 0 for t, _, w in rig.velocities if 0.4 <= t < 0.8)
    assert all(v == w == 0 for t, v, w in rig.velocities if t >= 0.8)
    assert not result.success and rig.goal.state == '취소'

    previous_count = len(rig.velocities)
    next_goal = Goal()
    rig.clock.on_step = lambda now: rig.node.obj_callback(detection(class_id=-1))
    assert rig.node.goal_callback(None) == tracking.GoalResponse.ACCEPT
    result = rig.node.execute_callback(next_goal)
    assert not result.success and next_goal.state == '실패'
    assert all(v == w == 0 for _, v, w in rig.velocities[previous_count:])


@pytest.mark.parametrize('kind,expected,limit', [
    ('정렬', '정렬 시간 초과', 15.0),
    ('접근', '전체 주행 시간 초과', 40.0),
])
def test_repeated_reacquisition_cannot_extend_deadlines(rig, kind, expected, limit):
    """재탐지에 반복 성공해도 진행 중인 정렬과 전체 수거 시간을 늘리지 않는다."""
    def feed(now):
        step = round(now * 10) % 10
        x = 250.0 if kind == '정렬' else (330.0 if step in (2, 3) else 350.0)
        rig.node.obj_callback(detection(x=x, class_id=-1 if step in (4, 5, 6) else 0))

    rig.clock.on_step = feed
    result = rig.run()
    assert not result.success and expected in result.message
    assert rig.clock.now <= limit + 0.1
    assert sum('제자리 재탐지' in log for log in rig.logs) > 2
    assert rig.velocities[-1][1:] == (0, 0)


def test_close_but_misaligned_target_must_realign_before_final_motion(rig):
    """물체가 가까워도 마지막 전진 전에 좌우 정렬을 완료한다."""
    rig.clock.on_step = lambda now: rig.node.obj_callback(detection(
        x=350.0 if now < 0.3 else 330.0, bottom=440.0))
    result = rig.run()
    assert not result.success and '정렬 시간 초과' in result.message
    assert not any(v > 0 for _, v, _ in rig.velocities)
    assert not any('마지막 수거 전진' in log for log in rig.logs)


def test_final_motion_runs_once_even_when_target_leaves_view(rig):
    """화면 아래로 사라진 뒤에도 마지막 전진 시간을 다시 세지 않는다."""
    def feed(now):
        rig.node.obj_callback(detection(bottom=440.0, class_id=0 if now < 0.6 else -1))

    rig.clock.on_step = feed
    result = rig.run()
    moving = [t for t, v, _ in rig.velocities if v > 0]
    assert result.success
    assert 2.9 <= rig.velocities[-1][0] - moving[0] <= 3.1
    assert sum('마지막 수거 전진' in log for log in rig.logs) == 1


def test_cancel_during_final_motion_prevents_later_nonzero(rig):
    """마지막 전진 중 취소해도 이후 이동 명령을 내보내지 않는다."""
    def feed(now):
        rig.node.obj_callback(detection(bottom=440.0))
        if now == 1.0:
            rig.goal.is_cancel_requested = True
            rig.node.cancel_callback(rig.goal)

    rig.clock.on_step = feed
    result = rig.run()
    assert not result.success and result.message == 'STOP'
    assert rig.goal.state == '취소'
    assert any(v > 0 for t, v, _ in rig.velocities if t < 1.0)
    assert all(v == w == 0 for t, v, w in rig.velocities if t >= 1.0)


def test_total_deadline_also_applies_to_final_motion(rig):
    """마지막 전진에도 전체 주행 제한 시간을 적용한다."""
    rig.node.tracking_timeout_sec = 2.0
    rig.clock.on_step = lambda now: rig.node.obj_callback(detection(bottom=440.0))
    result = rig.run()
    assert not result.success and '전체 주행 시간 초과' in result.message
    assert rig.clock.now <= 2.1


def test_late_on_response_is_followed_by_off_before_another_goal(rig):
    """늦은 켜기 응답 뒤 끄기를 완료하기 전까지 다음 수거를 거절한다."""
    rig.client.pending_on = rig.client.pending_off = True
    result = rig.run()
    assert not result.success and rig.clock.now <= 6.1
    assert [enable for enable, _ in rig.requests] == [True]
    assert rig.node.goal_callback(None) == tracking.GoalResponse.REJECT
    rig.requests[0][1].set_result(SimpleNamespace(success=True))
    assert [enable for enable, _ in rig.requests] == [True, False]
    assert rig.node.goal_callback(None) == tracking.GoalResponse.REJECT
    count = len(rig.velocities)
    rig.requests[-1][1].set_result(SimpleNamespace(success=True))
    assert rig.node.goal_callback(None) == tracking.GoalResponse.ACCEPT
    assert len(rig.velocities) == count


@pytest.mark.parametrize('pending_off', [False, True])
def test_cancel_before_execution_never_enables_tracking(rig, pending_off):
    """실행 전 취소와 중복 수거 요청이 추적을 다시 켜지 않게 한다."""
    rig.client.pending_off = pending_off
    assert rig.node.goal_callback(None) == tracking.GoalResponse.ACCEPT
    assert rig.node.goal_callback(None) == tracking.GoalResponse.REJECT
    rig.goal.is_cancel_requested = True
    rig.node.cancel_callback(rig.goal)
    result = rig.node.execute_callback(rig.goal)
    assert not result.success and rig.goal.state == '취소'
    assert [enable for enable, _ in rig.requests] == [False]
    assert not any(v or w for _, v, w in rig.velocities)
    if pending_off:
        assert rig.node.goal_callback(None) == tracking.GoalResponse.REJECT
        rig.requests[-1][1].set_result(SimpleNamespace(success=True))
    assert rig.node.goal_callback(None) == tracking.GoalResponse.ACCEPT


def test_pending_cleanup_preserves_original_failure_and_stops_first(rig):
    """끄기 응답이 늦어도 정지하고 원래 실패 원인을 유지한다."""
    rig.client.pending_off = True
    result = rig.run()
    assert not result.success
    assert '재탐지 시간 초과' in result.message
    assert '추적 모드 해제 미확인' in result.message
    assert rig.clock.now <= 7.1
    assert all(v == w == 0 for _, v, w in rig.velocities)
    assert rig.node.goal_callback(None) == tracking.GoalResponse.REJECT


@pytest.mark.parametrize('stop_reason', [None, 'STOP', 'BATTERY_LOW'])
def test_autonav_failure_closes_servo_and_selects_patrol_or_home(monkeypatch, stop_reason):
    """실패 시 서보를 정리하고 STOP 여부에 따라 순찰 또는 HOME을 선택한다."""
    node = object.__new__(auto_nav.AutoNav)
    actions = []
    node.get_logger = lambda: SimpleNamespace(warn=lambda text: actions.append(('로그', text)))
    node.trigger_servo_movement = lambda a, b: actions.append(('서보', a, b))
    node.send_goal = lambda x, y: actions.append(('순찰', x, y))
    node.return_home_by_stop = lambda: actions.append(('HOME',))
    node.cancel_reason, node.stop_pending = stop_reason, False
    node.resume_x, node.resume_y, node.object_found = 1.2, 3.4, True
    monkeypatch.setattr(auto_nav, 'time', SimpleNamespace(monotonic=lambda: 100.0))
    future = SimpleNamespace(result=lambda: SimpleNamespace(
        status=auto_nav.GoalStatus.STATUS_ABORTED,
        result=SimpleNamespace(success=False, message='정렬 시간 초과')))
    node.recycle_tracking_result_callback(future)
    assert ('서보', 0, 0) in actions
    if stop_reason:
        assert ('HOME',) in actions and not any(x[0] == '순찰' for x in actions)
    else:
        assert ('순찰', 1.2, 3.4) in actions
        assert node.tracking_retry_after == 102.0 and not node.object_found
