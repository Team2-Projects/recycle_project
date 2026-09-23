"""하역·HOME 복귀 중 추론 중지를 갱신하고, ON 응답 뒤 순찰을 연결한다."""

import time

from std_srvs.srv import SetBool


class InferenceControl:
    """응답 유실·YOLO 재시작에 대비해 원하는 상태를 주기적으로 전송한다."""

    def __init__(self, node):
        self.node = node
        self.client = node.create_client(SetBool, 'set_inference_enabled')
        self.enabled = True
        self.confirmed = None
        self._generation = 0
        self._future = None
        self._on_enabled = None
        self._next_request = 0.0
        self._last_warning = None
        self.timer = node.create_timer(0.5, self._poll)

    @property
    def ready(self):
        """현재 요청에 대한 ON 응답이 확인되었는지 반환한다."""
        return self.enabled and self.confirmed is True

    def set_enabled(self, enabled, on_enabled=None):
        """OFF는 이동을 막지 않으며 ON 후속 동작은 서버 확인 뒤 한 번 실행한다."""
        if enabled != self.enabled:
            self._generation += 1
            self.confirmed = None
        self.enabled = enabled
        self._on_enabled = on_enabled if enabled else None
        self._next_request = 0.0
        if self.ready:
            self._run_continuation()
        self._poll()

    def _warn(self, message):
        now = time.monotonic()
        if self._last_warning is None or now - self._last_warning >= 5.0:
            self.node.get_logger().warning(f'YOLO 추론 제어: {message} (자동 재시도)')
            self._last_warning = now

    def _poll(self):
        now = time.monotonic()
        if self._future is not None:
            if now < self._deadline:
                return
            future, self._future = self._future, None
            self.client.remove_pending_request(future)
            self.confirmed = None
            self._warn('응답 시간 초과')
        if now < self._next_request:
            return
        self._next_request = now + 1.0
        if not self.client.service_is_ready():
            self.confirmed = None
            self._warn('set_inference_enabled 서비스 대기 중')
            return
        request = SetBool.Request()
        request.data = self.enabled
        generation = self._generation
        try:
            self._future = self.client.call_async(request)
        except Exception as exc:
            self.confirmed = None
            self._warn(str(exc))
            return
        self._deadline = now + 3.0
        self._future.add_done_callback(
            lambda future: self._response(future, request.data, generation))

    def _response(self, future, enabled, generation):
        if future is not self._future:
            return  # 타임아웃으로 폐기한 응답
        self._future = None
        if generation != self._generation:
            # OFF 처리 중 순찰 재개 요청이 오면 OFF 확인 뒤 ON을 보낸다.
            self._next_request = 0.0
            self._poll()
            return
        try:
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(getattr(response, 'message', '빈 응답'))
        except Exception as exc:
            self.confirmed = None
            self._warn(str(exc))
            return
        changed = self.confirmed != enabled
        self.confirmed = enabled
        if changed:
            self.node.get_logger().info(f'YOLO 추론 {"ON" if enabled else "OFF"} 확인')
        if self.ready:
            self._run_continuation()

    def _run_continuation(self):
        callback, self._on_enabled = self._on_enabled, None
        if callback is not None:
            callback()
