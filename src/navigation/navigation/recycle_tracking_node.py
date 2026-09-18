"""영상의 좌우 오차를 보정하며 접근하고, 제한 시간 안에 수거 시도를 끝낸다."""

import math
import time
from threading import Event, Lock

from geometry_msgs.msg import Twist

from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking

from navigation_interface.action import RecycleActionMsg

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class RecycleTrackingNode(Node):
    """작은 오차는 이동 중 보정하고 큰 오차는 정지한 뒤 재정렬한다."""

    def __init__(self):
        """기존 수거 통신과 정렬·접근에 필요한 제한값을 준비한다."""
        super().__init__('recycle_tracking_node')
        defaults = {
            'align_tolerance_px': 10.0,
            'realign_threshold_px': 35.0,
            'align_timeout_sec': 15.0,
            'tracking_timeout_sec': 40.0,
            'detection_timeout_sec': 1.0,
            'target_lost_timeout_sec': 4.0,
            'tracking_service_timeout_sec': 3.0,
            'approach_steer_kp': 0.0004,
            'approach_max_angular_speed': 0.06,
        }
        for name, default in defaults.items():
            value = float(self.declare_parameter(name, default).value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: 0보다 큰 유한한 값이 필요합니다.')
            setattr(self, name, value)
        if self.realign_threshold_px <= self.align_tolerance_px:
            raise ValueError('재정렬 시작 오차는 정렬 완료 오차보다 커야 합니다.')

        self.cancel_event = Event()
        self._lock = Lock()
        self._busy = False
        self._observation = (None, 0.0, 0)
        self._tracking_future = None
        self._tracking_off_confirmed = Event()
        self._tracking_off_confirmed.set()
        self.cb_group = ReentrantCallbackGroup()
        self.sub = self.create_subscription(
            DetectedObject, '/classified_detected_object_info',
            self.obj_callback, 1, callback_group=self.cb_group)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tracking_cli = self.create_client(
            SetTracking, 'set_tracking_mode', callback_group=self.cb_group)
        self._action_server = ActionServer(
            self, RecycleActionMsg, 'recycle_tracking_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group)
        self.get_logger().info(
            f'수거 제어 설정: 정렬 완료 {self.align_tolerance_px:g}픽셀 이내, '
            f'재정렬 시작 {self.realign_threshold_px:g}픽셀 이상, '
            f'정렬 제한 {self.align_timeout_sec:g}초, '
            f'재탐지 제한 {self.target_lost_timeout_sec:g}초, '
            f'전체 주행 제한 {self.tracking_timeout_sec:g}초')

    def obj_callback(self, msg):
        """수신 시각과 번호를 함께 저장해 같은 검출 결과를 중복 집계하지 않는다."""
        with self._lock:
            self._observation = (msg, time.monotonic(), self._observation[2] + 1)

    def goal_callback(self, _request):
        """이전 수거 또는 추적 모드 해제가 끝나기 전에는 새 수거를 받지 않는다."""
        with self._lock:
            if self._busy or not self._tracking_off_confirmed.is_set():
                self.get_logger().warn('수거 요청 거절: 이전 수거 또는 추적 모드 해제 대기 중')
                return GoalResponse.REJECT
            self._busy = True
            self.cancel_event.clear()
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        """취소 이후 제어 루프가 다시 전진 명령을 보내지 않도록 함께 잠근다."""
        with self._lock:
            self.cancel_event.set()
            self.cmd_vel_pub.publish(Twist())
        self.get_logger().info('수거 취소 요청: 즉시 정지')
        return CancelResponse.ACCEPT

    def _publish_velocity(self, linear=0.0, angular=0.0):
        with self._lock:
            msg = Twist()
            if not self.cancel_event.is_set():
                msg.linear.x = linear
                msg.angular.z = angular
            self.cmd_vel_pub.publish(msg)

    def _enable_tracking(self, goal_handle, deadline):
        if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
            return False
        request = SetTracking.Request()
        request.enable = True
        self._tracking_off_confirmed.clear()
        self._tracking_future = self.tracking_cli.call_async(request)
        end = min(deadline, time.monotonic() + self.tracking_service_timeout_sec)
        while rclpy.ok():
            if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                return False
            if self._tracking_future.done():
                try:
                    response = self._tracking_future.result()
                    return response is not None and response.success
                except Exception as exc:  # noqa: B902
                    self.get_logger().error(f'추적 모드 켜기 응답 오류: {exc}')
                    return False
            if time.monotonic() >= end:
                self.get_logger().warn('추적 모드 켜기 응답 시간 초과')
                return False
            self.cancel_event.wait(0.05)
        return False

    def _disable_tracking(self):
        """늦은 켜기 응답 뒤에 끄기를 보내고, 응답 대기만 제한 시간 안에 끝낸다."""
        self._tracking_off_confirmed.clear()
        completed = Event()

        def disabled(future):
            try:
                response = future.result()
                if response is not None and response.success:
                    self._tracking_off_confirmed.set()
                else:
                    self.get_logger().error('추적 모드 해제 실패: 새 수거 요청을 보류합니다.')
            except Exception as exc:  # noqa: B902
                self.get_logger().error(f'추적 모드 해제 오류: {exc}')
            finally:
                completed.set()

        def send_off(_future=None):
            try:
                request = SetTracking.Request()
                request.enable = False
                self._tracking_future = self.tracking_cli.call_async(request)
                self._tracking_future.add_done_callback(disabled)
            except Exception as exc:  # noqa: B902
                self.get_logger().error(f'추적 모드 해제 요청 오류: {exc}')
                completed.set()

        pending = self._tracking_future
        if pending is not None and not pending.done():
            pending.add_done_callback(send_off)
        else:
            send_off()
        end = time.monotonic() + self.tracking_service_timeout_sec
        while rclpy.ok() and time.monotonic() < end:
            if completed.wait(timeout=0.05):
                break
        if not self._tracking_off_confirmed.is_set():
            self.get_logger().warn('추적 모드 해제 미확인: 정지 상태로 종료하고 새 수거를 보류합니다.')
        return self._tracking_off_confirmed.is_set()

    def execute_callback(self, goal_handle):
        """성공·실패·취소 모두 정지를 먼저 처리한 뒤 기존 Action 결과로 알린다."""
        started = time.monotonic()
        deadline = started + self.tracking_timeout_sec
        success, reason = False, '추적 모드 켜기 실패'
        self.get_logger().info('수거 정렬·접근 시작')
        try:
            if self._enable_tracking(goal_handle, deadline):
                success, reason = self._run_tracking(goal_handle, deadline)
        except Exception as exc:  # noqa: B902
            reason = f'정렬·접근 처리 오류: {exc}'
            self.get_logger().error(reason)
        finally:
            # 서비스가 늦어져도 마지막 이동 명령이 유지되지 않도록 먼저 정지한다.
            self._publish_velocity()
            cleanup_ok = self._disable_tracking()

        result = RecycleActionMsg.Result()
        if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
            result.success, result.message = False, 'STOP'
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
        else:
            result.success = success
            result.message = reason
            if not cleanup_ok:
                result.message += ' (추적 모드 해제 미확인)'
            if success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        self.get_logger().info(
            f'수거 정렬·접근 종료: {result.message}, 경과 {time.monotonic() - started:.1f}초')
        with self._lock:
            self._busy = False
        return result

    @staticmethod
    def _valid_detection(msg):
        if msg is None or msg.id < 0:
            return False
        return (all(math.isfinite(value) for value in msg.coord)
                and msg.coord[2] > 0 and msg.coord[3] > 0)

    def _run_tracking(self, goal_handle, deadline):
        # 좌표·속도·마지막 전진 시간은 main에서 사용하던 보정값을 유지한다.
        reference_x, stop_lower_y = 350.0, 430.0
        forward_speed, align_speed = 0.10, 0.05
        period, final_duration = 0.10, 3.0
        phase = '정렬'
        align_deadline = time.monotonic() + self.align_timeout_sec
        aligned_count, close_count = 0, 0
        lost_since, final_until = None, None
        lost_mode = None
        # 새 수거 요청에 담긴 검출 위치로 시작하고, 이후 유효한 검출로 갱신한다.
        target = goal_handle.request
        last_error = None
        if (math.isfinite(target.target_x) and math.isfinite(target.target_h)
                and target.target_h > 0):
            last_error = reference_x - target.target_x
        with self._lock:
            start_sequence = self._observation[2]
        seen_sequence = start_sequence
        self.get_logger().info('정렬 시작: 새 검출 결과를 기다립니다.')

        while rclpy.ok():
            now = time.monotonic()
            if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                return False, 'STOP'
            # 재정렬이나 재검출이 반복되어도 전체 주행 제한 시간은 갱신하지 않는다.
            if now >= deadline:
                return False, '수거 시도 전체 주행 시간 초과'
            if phase == '정렬' and now >= align_deadline:
                return False, '정렬 시간 초과'
            if phase == '최종 전진':
                # 수거 직전에는 물체가 화면 아래로 사라질 수 있다. 한 번만 전진한다.
                if final_until is None:
                    final_until = now + final_duration
                if now >= final_until:
                    return True, '정렬 및 접근 완료'
                self._publish_velocity(forward_speed)
                self.cancel_event.wait(period)
                continue

            with self._lock:
                msg, received, sequence = self._observation
            fresh = sequence > start_sequence and now - received <= self.detection_timeout_sec
            new_detection = sequence != seen_sequence
            seen_sequence = sequence
            if not fresh or not self._valid_detection(msg):
                aligned_count, close_count = 0, 0
                if lost_since is None:
                    lost_since = now
                if now - lost_since >= self.target_lost_timeout_sec:
                    return False, '대상 재탐지 시간 초과'
                angular = 0.0
                mode = '유효한 검출 좌표 없음: 정지 후 재탐지 대기'
                if not fresh:
                    mode = '새 검출 정보 없음: 정지 후 수신 대기'
                elif (msg.id == -1 and last_error is not None
                      and abs(last_error) > self.align_tolerance_px):
                    # 미검출 메시지의 0 좌표가 아닌 마지막 유효 위치로 방향을 정한다.
                    angular = math.copysign(align_speed, last_error)
                    direction = '왼쪽' if angular > 0 else '오른쪽'
                    mode = f'대상 미검출: 마지막 검출 위치인 {direction}으로 제자리 재탐지'
                    if phase != '정렬':
                        phase = '정렬'
                        align_deadline = now + self.align_timeout_sec
                if mode != lost_mode:
                    self.get_logger().info(mode)
                    lost_mode = mode
                self._publish_velocity(angular=angular)
                self.cancel_event.wait(period)
                continue
            if lost_since is not None:
                self.get_logger().info('대상 재검출: 정렬·접근 재개')
                lost_since = None
                lost_mode = None

            error = reference_x - msg.coord[0]
            last_error = error
            lower_y = msg.coord[1] + msg.coord[3] / 2.0
            if phase == '정렬':
                if abs(error) <= self.align_tolerance_px:
                    self._publish_velocity()
                    if new_detection:
                        aligned_count += 1
                    if aligned_count >= 2:
                        phase = '접근'
                        self.get_logger().info(f'정렬 완료: 좌우 오차 {error:.1f}픽셀, 접근 시작')
                else:
                    aligned_count = 0
                    self._publish_velocity(angular=math.copysign(align_speed, error))
            elif (abs(error) >= self.realign_threshold_px
                  or (lower_y >= stop_lower_y and abs(error) > self.align_tolerance_px)):
                self._publish_velocity()
                phase = '정렬'
                align_deadline = now + self.align_timeout_sec
                aligned_count, close_count = 0, 0
                self.get_logger().info(f'재정렬 시작: 좌우 오차 {error:.1f}픽셀, 전진 정지')
            elif lower_y >= stop_lower_y:
                self._publish_velocity()
                if new_detection:
                    close_count += 1
                if close_count >= 2:
                    phase = '최종 전진'
                    self.get_logger().info('접근 기준 도달: 마지막 수거 전진 3초 시작')
            else:
                close_count = 0
                angular = 0.0
                if abs(error) > self.align_tolerance_px:
                    angular = max(-self.approach_max_angular_speed, min(
                        self.approach_max_angular_speed, self.approach_steer_kp * error))
                self._publish_velocity(forward_speed, angular)
            self.cancel_event.wait(period)
        return False, '노드 종료로 수거 중단'


def main(args=None):
    """검출·취소 응답을 처리하면서 수거 노드를 실행한다."""
    rclpy.init(args=args)
    node = RecycleTrackingNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('종료 요청: 수거 노드를 정지합니다.')
    finally:
        node.cancel_event.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
