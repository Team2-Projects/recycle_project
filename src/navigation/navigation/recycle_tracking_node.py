"""ROS 2 action adapter for the bounded alignment/approach controller."""

from dataclasses import fields
from threading import Event, Lock, RLock
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from navigation_interface.action import RecycleActionMsg

from navigation.tracking_control import (
    Command, Observation, Phase, TrackingConfig, TrackingController,
)


class RecycleTrackingNode(Node):
    def __init__(self):
        super().__init__('recycle_tracking_node')
        defaults = TrackingConfig()
        values = {}
        for field in fields(defaults):
            self.declare_parameter(
                field.name, getattr(defaults, field.name),
                ParameterDescriptor(read_only=True, description='Edit YAML and restart this node'),
            )
            values[field.name] = self.get_parameter(field.name).value
        self.config = TrackingConfig(**values)

        self.cancel_event = Event()
        self.shutdown_event = Event()
        self._state_lock = RLock()
        self._goal_lock = Lock()
        self._goal_busy = False
        self._controller = None
        self._object_msg_seq = 0
        # Serialized object callbacks preserve callback-order frame counting.
        # Service responses/cancel requests can execute while the action waits.
        self._object_group = MutuallyExclusiveCallbackGroup()
        self._action_group = ReentrantCallbackGroup()
        self._service_group = ReentrantCallbackGroup()
        self.sub = self.create_subscription(
            DetectedObject, '/classified_detected_object_info', self.obj_callback,
            1, callback_group=self._object_group,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tracking_cli = self.create_client(
            SetTracking, 'set_tracking_mode', callback_group=self._service_group,
        )
        self._action_server = ActionServer(
            self, RecycleActionMsg, 'recycle_tracking_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._action_group,
        )
        self.get_logger().info(
            f'Recycle tracking: x={self.config.align_reference_x:.0f}, '
            f'period={self.config.control_period_sec:.2f}s, '
            f'stop_only_test_mode={self.config.stop_only_test_mode}'
        )

    def _publish_command(self, command=Command()):
        """Caller owns _state_lock when motion is active."""
        msg = Twist()
        if not self.cancel_event.is_set() and not self.shutdown_event.is_set():
            msg.linear.x = float(command.linear_x)
            msg.angular.z = float(command.angular_z)
        self.cmd_vel_pub.publish(msg)

    def goal_callback(self, request):
        with self._goal_lock:
            if self._goal_busy or self.shutdown_event.is_set() or request.index < 0:
                return GoalResponse.REJECT
            self._goal_busy = True
            # Clear here, not in execute_callback: an early cancel must survive.
            self.cancel_event.clear()
        return GoalResponse.ACCEPT

    def obj_callback(self, msg):
        with self._state_lock:
            self._object_msg_seq += 1
            controller = self._controller
            if controller is None or controller.done:
                return
            now = time.monotonic()
            try:
                x, y, width, height = (float(value) for value in msg.coord)
                obs = Observation(
                    self._object_msg_seq, now, int(msg.id), float(msg.confidence),
                    x, y, width, height,
                )
            except (TypeError, ValueError, AttributeError, OverflowError):
                obs = Observation(self._object_msg_seq, now, -1, 0.0, 0.0, 0.0, 0.0, 0.0)
            previous_phase = controller.phase
            controller.observe(obs)
            command = controller.step(now)
            # Brake on invalid/near-threshold results without waiting for the
            # next control tick. Never publish a nonzero velocity from here.
            entering_realign = (previous_phase == Phase.APPROACH
                                and controller.phase == Phase.REALIGN)
            if command == Command() or entering_realign:
                self._publish_command()

    def cancel_callback(self, goal_handle):
        with self._state_lock:
            self.cancel_event.set()
            if self._controller is not None:
                self._controller.cancel()
            self._publish_command()
        self.get_logger().warn('Tracking cancel 요청: 정지 명령 발행')
        return CancelResponse.ACCEPT

    def _running(self):
        return rclpy.ok() and not self.shutdown_event.is_set()

    def call_tracking_srv(self, enable, target_class_id=-1, honor_cancel=True):
        """Return (success, reason) with bounded discovery/response waiting."""
        deadline = time.monotonic() + self.config.tracking_service_timeout_sec
        future = None
        try:
            while self._running():
                if honor_cancel and self.cancel_event.is_set():
                    return False, 'CANCELED'
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.get_logger().error(f'SetTracking({enable}): 서비스 준비 시간 초과')
                    return False, 'SERVICE_TIMEOUT'
                if self.tracking_cli.wait_for_service(timeout_sec=min(0.05, remaining)):
                    break
            else:
                return False, 'SHUTDOWN'
            request = SetTracking.Request()
            request.enable = bool(enable)
            request.target_class_id = int(target_class_id if enable else -1)
            future = self.tracking_cli.call_async(request)
            ready = Event()
            future.add_done_callback(lambda _: ready.set())
            while self._running():
                if honor_cancel and self.cancel_event.is_set():
                    return False, 'CANCELED'
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.get_logger().error(f'SetTracking({enable}): 응답 시간 초과')
                    return False, 'SERVICE_TIMEOUT'
                if ready.wait(timeout=min(0.05, remaining)):
                    response = future.result()
                    if response is None:
                        self.get_logger().error(f'SetTracking({enable}): 빈 응답')
                        return False, 'SERVICE_ERROR'
                    reason = str(getattr(response, 'reason', '') or '').strip() or (
                        'OK' if response.success else 'SERVICE_ERROR'
                    )
                    if not response.success:
                        self.get_logger().error(f'SetTracking({enable}): 실패 응답 ({reason})')
                        return False, reason
                    return True, reason
            return False, 'SHUTDOWN'
        except Exception as exc:
            self.get_logger().error(f'SetTracking({enable}) 예외: {exc}')
            return False, 'SERVICE_ERROR'
        finally:
            if future is not None and not future.done():
                # Cancels only the local future; server-side work is not revoked.
                future.cancel()

    def _run_control(self, goal_handle):
        with self._state_lock:
            # Discard the goal's old x/y and all pre-service detections.
            self._controller = TrackingController(
                self.config, int(goal_handle.request.index), time.monotonic(),
            )
        last_status = None
        next_tick = time.monotonic()
        while self._running():
            with self._state_lock:
                controller = self._controller
                if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                    controller.cancel()
                command = controller.step(time.monotonic())
                self._publish_command(command)
                status = controller.status
                done = controller.done
                success = controller.phase == Phase.SUCCEEDED
                reason = controller.reason
            if status != last_status:
                self.get_logger().info(f'Tracking: {status}')
                # RecycleActionMsg.Feedback 에 status 필드가 없는 버전과도
                # 호환되게 한다. 없으면 로그만 남기고 피드백은 생략한다.
                feedback = RecycleActionMsg.Feedback()
                if hasattr(feedback, 'status'):
                    feedback.status = status
                    goal_handle.publish_feedback(feedback)
                last_status = status
            if done:
                return success, reason
            next_tick += self.config.control_period_sec
            delay = next_tick - time.monotonic()
            if delay > 0:
                self.cancel_event.wait(delay)
            else:
                # Do not issue a burst of catch-up commands after a scheduling stall.
                next_tick = time.monotonic()
        return False, 'SHUTDOWN: 추적 노드 종료'

    def execute_callback(self, goal_handle):
        success = False
        message = 'INTERNAL_ERROR: 추적이 정상 종료되지 않음'
        cleanup_ok = False
        try:
            with self._state_lock:
                self._publish_command()
            if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                message = 'STOP'
            else:
                tracking_ok, tracking_reason = self.call_tracking_srv(
                    True, int(goal_handle.request.index)
                )
                if not tracking_ok:
                    if tracking_reason == 'VISION_NOT_READY':
                        message = 'VISION_NOT_READY: YOLO 결과 스트림이 아직 준비되지 않음'
                    else:
                        message = f'SERVICE_ERROR: YOLO 추적 모드 활성화 실패 ({tracking_reason})'
                else:
                    success, message = self._run_control(goal_handle)
        except Exception as exc:
            self.get_logger().error(f'Tracking 예외: {exc}')
            message = f'INTERNAL_ERROR: {type(exc).__name__}: {exc}'
            success = False
        finally:
            # Stop BEFORE a potentially slow service call on every exit path.
            with self._state_lock:
                self._controller = None
                try:
                    self._publish_command()
                except Exception as exc:
                    self.get_logger().error(f'정지 명령 발행 실패: {exc}')
            cleanup_ok, cleanup_reason = self.call_tracking_srv(False, -1, honor_cancel=False)

        try:
            result = RecycleActionMsg.Result()
            canceled = self.cancel_event.is_set() or goal_handle.is_cancel_requested
            if canceled:
                success = False
                message = 'STOP'
            elif not cleanup_ok and message.startswith('SENSOR_STALE:'):
                # Keep the primary fault identifiable. A guarded new action will
                # explicitly re-establish tracking mode before moving again.
                success = False
                message += '; CLEANUP_UNCONFIRMED: 추적 모드 해제 응답 없음; 재개 전 서비스 재확인'
            elif not cleanup_ok:
                success = False
                message = (
                    f'SERVICE_ERROR: YOLO 추적 모드 해제 확인 실패 ({cleanup_reason}); {message}'
                )
            result.success = success
            result.message = message
            # Only call canceled() once ROS has actually entered CANCELING.
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            elif success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            self.get_logger().info(f'Tracking result: {message}')
            return result
        finally:
            with self._goal_lock:
                self._goal_busy = False

    def request_shutdown(self):
        self.shutdown_event.set()
        self.cancel_event.set()
        with self._state_lock:
            if self._controller is not None:
                self._controller.cancel()
            try:
                self._publish_command()
            except Exception:
                pass  # Context may already be invalid after a ROS signal handler.


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = RecycleTrackingNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.request_shutdown()
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
