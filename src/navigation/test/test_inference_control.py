"""응답 유실과 재시작 중에도 OFF 갱신 및 ON 확인을 올바르게 처리하는지 검증한다."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from navigation import inference_control

import pytest


@pytest.fixture
def control(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(inference_control, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    requests = []

    def call_async(request):
        future = Future()
        requests.append((request.data, future))
        return future

    client = Mock()
    client.service_is_ready.return_value = True
    client.call_async.side_effect = call_async
    node = Mock()
    node.create_client.return_value = client
    controller = inference_control.InferenceControl(node)
    return SimpleNamespace(controller=controller, clock=clock, client=client,
                           requests=requests, node=node)


def respond(rig, success=True):
    rig.requests[-1][1].set_result(SimpleNamespace(success=success, message='test'))


def test_pause_refresh_and_detector_restart_receive_current_state(control):
    rig = control
    rig.controller.set_enabled(False)
    respond(rig)
    for now in (1.0, 2.0):
        rig.clock.now = now
        rig.controller._poll()
        respond(rig)
    assert [enabled for enabled, _ in rig.requests] == [False, False, False]
    rig.client.service_is_ready.return_value = False
    rig.clock.now = 3.0
    rig.controller._poll()
    assert rig.controller.confirmed is None
    rig.client.service_is_ready.return_value = True
    rig.clock.now = 4.0
    rig.controller._poll()
    assert not rig.requests[-1][0]
    respond(rig)
    assert rig.controller.confirmed is False


def test_lost_off_response_retries_on_and_ignores_late_off_response(control):
    rig = control
    resume = Mock()
    rig.controller.set_enabled(False)
    old = rig.requests[-1][1]
    rig.controller.set_enabled(True, resume)
    rig.clock.now = 3.01
    rig.controller._poll()
    rig.client.remove_pending_request.assert_called_once_with(old)
    assert rig.requests[-1][0]
    old.set_result(SimpleNamespace(success=True))
    assert not rig.controller.ready
    resume.assert_not_called()
    respond(rig)
    resume.assert_called_once()
    assert rig.controller.ready
    rig.clock.now = 4.1
    rig.controller._poll()
    respond(rig)
    resume.assert_called_once()


@pytest.mark.parametrize('error', ['rejected', 'exception', 'send_exception'])
def test_enable_failure_holds_patrol_until_successful_retry(control, error):
    rig = control
    resume = Mock()
    if error == 'send_exception':
        rig.client.call_async.side_effect = RuntimeError('send failed')
    rig.controller.set_enabled(True, resume)
    if error == 'rejected':
        respond(rig, success=False)
    elif error == 'exception':
        rig.requests[-1][1].set_exception(RuntimeError('response failed'))
    assert not rig.controller.ready
    resume.assert_not_called()
    rig.clock.now = 1.0
    if error == 'send_exception':
        future = Future()
        rig.client.call_async.side_effect = lambda request: future
    rig.controller._poll()
    if error == 'send_exception':
        future.set_result(SimpleNamespace(success=True))
    else:
        respond(rig)
    resume.assert_called_once()


def test_canceling_continuation_prevents_late_enable_from_starting_patrol(control):
    rig = control
    resume = Mock()
    rig.controller.set_enabled(True, resume)
    rig.controller.set_enabled(True)
    respond(rig)
    assert rig.controller.ready
    resume.assert_not_called()
