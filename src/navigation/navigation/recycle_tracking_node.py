"""ROS 2 action adapter for the bounded alignment/approach controller."""

from dataclasses import fields
from threading import Event, Lock, RLock
import time
import math

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist
from navigation_interface.msg import CollisionCommand
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from navigation_interface.action import RecycleActionMsg

from navigation.tracking_control import (
    Command, Observation, Phase, TrackingConfig, TrackingController,
)

from navigation.collision_safety import (
    CollisionConfig, CollisionSafety, moving,
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
        collision_defaults = CollisionConfig()
        collision_values = {}
        for field in fields(collision_defaults):
            name = 'collision_' + field.name
            self.declare_parameter(name, getattr(collision_defaults, field.name),
                                   ParameterDescriptor(read_only=True))
            collision_values[field.name] = self.get_parameter(name).value
        self.collision_config = CollisionConfig(**collision_values)

        self.cancel_event = Event()
        self.shutdown_event = Event()
        self._state_lock = RLock()
        self._goal_lock = Lock()
        self._goal_busy = False
        self._controller = None
        self._owns_cmd_vel = False
        self._control_finished = False
        self._safety = CollisionSafety(self.collision_config)
        self._probe = Command()
        self._last_collision_status = None
        self._object_msg_seq = 0
        # One owner and one outstanding SetTracking call across ALL actions.
        # A timeout ends local waiting, never the server's work.
        self._mode_future = None
        self._mode_request_enable = None
        self._mode_cleanup_ready = True
        self._mode_retry_after = 0.0
        # Serialized object callbacks preserve callback-order frame counting.
        # Service responses/cancel requests can execute while the action waits.
        self._object_group = MutuallyExclusiveCallbackGroup()
        self._action_group = ReentrantCallbackGroup()
        self._service_group = ReentrantCallbackGroup()
        self._collision_group = MutuallyExclusiveCallbackGroup()
        self._watchdog_group = MutuallyExclusiveCallbackGroup()
        self.sub = self.create_subscription(
            DetectedObject, '/classified_detected_object_info', self.obj_callback,
            1, callback_group=self._object_group,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self._raw_pub = self.create_publisher(CollisionCommand, '/tracking_cmd_vel_raw', 1)
        self.create_subscription(CollisionCommand, '/tracking_cmd_vel_safe', self._safe_callback,
                                 1, callback_group=self._collision_group)
        # Watchdogs must continue even if ROS simulated time pauses.
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self.collision_config.watchdog_period_sec,
                          self._collision_watchdog, callback_group=self._watchdog_group,
                          clock=self._steady_clock)
        # Only settle the existing ON/OFF service while idle; no collision probes.
        self.create_timer(0.10, self._cleanup_tick,
                          callback_group=self._watchdog_group, clock=self._steady_clock)
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
            f'stop_only_test_mode={self.config.stop_only_test_mode}; '
            'collision_protocol=request_id_v1'
        )

    @staticmethod
    def _twist(command=Command()):
        msg = Twist()
        msg.linear.x = float(command.linear_x)
        msg.angular.z = float(command.angular_z)
        return msg

    def _publish_output(self, command=Command()):
        # This is the ONLY /cmd_vel publish site. Idle patrol is never written
        # to by this node, not even with zero watchdog heartbeats.
        if not self._owns_cmd_vel:
            return
        if self.cancel_event.is_set() or self.shutdown_event.is_set():
            command = Command()
        self.cmd_vel_pub.publish(self._twist(command))

    def _publish_command(self, command=Command(), force=False):
        """Send intent ONLY to Collision Monitor; zero may brake immediately."""
        if not self._owns_cmd_vel:
            return
        if (self.cancel_event.is_set() or self.shutdown_event.is_set()
                or self._safety.failure or self._control_finished):
            command = Command()
        phase = self._controller.phase.value if self._controller else 'IDLE'
        request_id = self._safety.request(command, phase, time.monotonic(), force=force)
        self._raw_pub.publish(CollisionCommand(request_id=request_id, velocity=self._twist(command)))
        if not moving(command):
            self._publish_output()

    def _set_collision_status(self, status):
        if self._last_collision_status != status:
            self._last_collision_status = status
            self.get_logger().info(f'Collision: {status}')

    def _fail_for_safety(self, reason):
        if self._control_finished:
            self._publish_output()
            return
        controller = self._controller
        if controller is not None and controller.phase != Phase.CANCELED:
            # Safety outranks a concurrent visual success/timeout. Otherwise a
            # generic APPROACH_TIMEOUT could make AutoNav restart patrol.
            controller.phase = Phase.FAILED
            cause = (self._safety.last_fault_reason
                     if reason in ('SAFETY_UNAVAILABLE', 'SAFETY_NOT_READY') else '')
            detail = f'; cause={cause}' if cause else ''
            controller.reason = (
                f'{reason}: 충돌 안전 경로 중단; AutoNav 복구 정책으로 전달{detail}')
        self._safety.fail(reason)
        self._publish_output()
        self._set_collision_status(reason)

    def _apply_safety_locked(self):
        # Call ONLY with _state_lock held. Take time HERE, after all request /
        # receive updates, not at the beginning of the outer control iteration.
        # A snapshot taken before request() creates negative raw age at 10 Hz.
        now = time.monotonic()
        if not self._owns_cmd_vel:
            return
        controller = self._controller
        if controller is None or self._control_finished:
            self._publish_output()
            return
        if self.cancel_event.is_set() or self.shutdown_event.is_set():
            self._publish_output()
            return
        decision = self._safety.evaluate(now)
        self._set_collision_status(decision.state)
        if decision.failure:
            self._fail_for_safety(decision.failure)
            return
        if controller.done:
            self._publish_output()
            return
        if decision.realign:
            self._publish_output()
            if controller.resume_after_safety(now):
                self._safety.acknowledge_realign(now)
                self._probe = Command()
                self._publish_command(force=True)
                self._set_collision_status('REALIGN_REQUIRED')
            else:
                self._fail_for_safety('FINAL_APPROACH_INTERRUPTED')
            return
        if decision.pause:
            # Freeze progress/near confirmation but keep a hypothetical motion
            # probe going ONLY to the monitor so it can detect obstacle removal.
            if not controller.safety_paused:
                self._probe = self._safety.raw
                controller.pause_for_safety()
            self._publish_output()
            return
        if controller.safety_paused:
            self._publish_output()
            return
        if controller.phase == Phase.FINAL_APPROACH and moving(decision.command):
            controller.arm_final_motion(now)
            self._safety.mark_final_started()
        # A changed/zero intent has already invalidated old safe samples. Never
        # let a late moving callback undo an immediate vision/cancel brake.
        self._publish_output(decision.command)

    def _safe_callback(self, msg):
        with self._state_lock:
            if not self._owns_cmd_vel:
                return
            if self._control_finished:
                self._publish_output()
                return
            try:
                velocity = msg.velocity
                numbers = (velocity.linear.x, velocity.linear.y, velocity.linear.z,
                           velocity.angular.x, velocity.angular.y, velocity.angular.z)
                if (not all(math.isfinite(v) for v in numbers)
                        or any(abs(v) > 1e-9 for v in (
                            velocity.linear.y, velocity.linear.z, velocity.angular.x, velocity.angular.y))):
                    raise ValueError('Invalid/nonplanar safe Twist')
                self._safety.accept_safe(Command(velocity.linear.x, velocity.angular.z), time.monotonic(),
                                         msg.request_id, msg.input_valid)
                self._apply_safety_locked()
            except Exception as exc:
                self.get_logger().error(f'Collision safe callback: {exc}')
                self._fail_for_safety('SAFETY_INVALID_OUTPUT')

    def _collision_watchdog(self):
        with self._state_lock:
            if not self._owns_cmd_vel:
                return  # No /cmd_vel heartbeats during Nav2 patrol.
            try:
                self._apply_safety_locked()
            except Exception as exc:
                self.get_logger().error(f'Collision watchdog: {exc}')
                self._fail_for_safety('SAFETY_INTERNAL_ERROR')

    def _cleanup_tick(self):
        with self._state_lock:
            if not self._goal_busy and not self.shutdown_event.is_set():
                self._poll_mode_cleanup(time.monotonic())

    def _await_safety_ready(self, goal_handle):
        deadline = time.monotonic() + self.collision_config.ready_timeout_sec
        while self._running() and time.monotonic() < deadline:
            if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                return False, 'STOP'
            with self._state_lock:
                self._publish_command()
                now = time.monotonic()
                if self._safety.failure:
                    return False, self._safety.failure
                if (self._safety.input_valid and self._safety.safe_at is not None
                        and 0 <= now - self._safety.safe_at <= self.collision_config.safe_timeout_sec
                        and self._safety.safe == Command()):
                    return True, ''
            self.cancel_event.wait(self.config.control_period_sec)
        return False, 'SAFETY_NOT_READY: Monitor scan/TF 또는 속도 응답 없음'

    def goal_callback(self, request):
        with self._state_lock, self._goal_lock:
            if (self._goal_busy or self.shutdown_event.is_set() or request.index < 0
                    or not self._mode_cleanup_ready or self._mode_future is not None):
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
            if (controller.reason.split(':', 1)[0] in ('ALIGN_TIMEOUT', 'APPROACH_TIMEOUT')
                    and self._safety.state == 'COLLISION_SLOWING'):
                # Preserve the cause before the zero brake replaces raw intent.
                self._fail_for_safety('COLLISION_BLOCKED')
            # Object callbacks may brake, but may not authorize/finish timed
            # final motion. Only the serialized control tick checks completion.
            if controller.safety_paused:
                return
            entering_realign = previous_phase == Phase.APPROACH and controller.phase == Phase.REALIGN
            entering_final = previous_phase != Phase.FINAL_APPROACH and controller.phase == Phase.FINAL_APPROACH
            if (controller.done or controller.phase == Phase.SENSOR_WAIT
                    or controller.lost_frames or entering_realign or entering_final
                    or (controller.phase == Phase.APPROACH and controller.close_frames > 0)):
                self._publish_command(force=entering_final)

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

    def _take_mode_response(self):
        """Under _state_lock; only a server response settles a mode request."""
        future = self._mode_future
        if future is None or not future.done():
            return None
        try:
            response = future.result()
        except Exception:
            return None  # An exceptional/canceled waiter is not a server acknowledgment.
        if response is None:
            return None
        enable = self._mode_request_enable
        success = bool(response.success)
        reason = str(getattr(response, 'reason', '') or '').strip() or (
            'OK' if success else 'SERVICE_ERROR')
        self._mode_future = None
        self._mode_request_enable = None
        self._mode_cleanup_ready = not enable and success
        return enable, success, reason

    def _start_mode_request(self, enable, target_class_id=-1):
        request = SetTracking.Request()
        request.enable = bool(enable)
        request.target_class_id = int(target_class_id if enable else -1)
        self._mode_cleanup_ready = False
        self._mode_request_enable = bool(enable)
        self._mode_future = self.tracking_cli.call_async(request)

    def _poll_mode_cleanup(self, now):
        """Idle, nonblocking cleanup; unresolved ON/OFF is never overtaken."""
        settled = self._take_mode_response()
        if settled is not None and not settled[1]:
            self._mode_retry_after = now + self.config.tracking_service_timeout_sec
        if (self._mode_future is not None or self._mode_cleanup_ready
                or now < self._mode_retry_after):
            return
        try:
            if self.tracking_cli.service_is_ready():
                self._start_mode_request(False)
        except Exception as exc:
            self._mode_retry_after = now + self.config.tracking_service_timeout_sec
            self.get_logger().warn(f'SetTracking OFF 재시도 대기: {exc}')

    def call_tracking_srv(self, enable, target_class_id=-1, honor_cancel=True):
        """Bound the action wait while retaining every unresolved server call."""
        deadline = time.monotonic() + self.config.tracking_service_timeout_sec
        with self._state_lock:
            if enable and (self._mode_future is not None or not self._mode_cleanup_ready):
                return False, 'CLEANUP_PENDING'
            if not enable:
                self._mode_cleanup_ready = False
        while self._running():
            if honor_cancel and self.cancel_event.is_set():
                return False, 'CANCELED'
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, 'SERVICE_TIMEOUT'
            try:
                with self._state_lock:
                    settled = self._take_mode_response()
                    if settled is not None and settled[0] == bool(enable):
                        return settled[1], settled[2]
                    if self._mode_future is None and self.tracking_cli.service_is_ready():
                        self._start_mode_request(enable, target_class_id)
                    future = self._mode_future
                ready = Event()
                if future is not None and not future.done():
                    future.add_done_callback(lambda _, event=ready: event.set())
                ready.wait(timeout=min(0.05, remaining))
            except Exception as exc:
                self.get_logger().error(f'SetTracking({enable}) 예외: {exc}')
                return False, 'SERVICE_ERROR'
        return False, 'SHUTDOWN'

    def _run_control(self, goal_handle):
        with self._state_lock:
            self._controller = TrackingController(
                self.config, int(goal_handle.request.index), time.monotonic(),
                defer_final_motion=True,
            )
            self._publish_command(force=True)
        last_status = None
        next_tick = time.monotonic()
        while self._running():
            with self._state_lock:
                controller = self._controller
                if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                    controller.cancel()
                # Evaluate faults/interventions BEFORE the final timer can
                # produce success. The same RLock is used by all callbacks.
                self._apply_safety_locked()
                command = controller.step(time.monotonic())
                collision_limited_timeout = (
                    controller.reason.split(':', 1)[0] in ('ALIGN_TIMEOUT', 'APPROACH_TIMEOUT')
                    and self._safety.state == 'COLLISION_SLOWING'
                )
                if controller.done and (self._safety.suspended or collision_limited_timeout):
                    reason = ('COLLISION_BLOCKED' if (
                        self._safety.hold_started_at is not None or collision_limited_timeout)
                        else 'SAFETY_UNAVAILABLE')
                    self._fail_for_safety(reason)
                if not controller.done and controller.safety_paused:
                    # Probe a hypothetical direction to the private monitor;
                    # output remains strictly zero until fresh visual REALIGN.
                    command = self._probe
                if controller.done:
                    # Commit outcome under the same lock used by callbacks.
                    # Cleanup cannot resume movement or turn this terminal
                    # result into a later idle timeout. Explicit cancel wins.
                    self._control_finished = True
                    self._publish_command()
                else:
                    self._publish_command(command)
                    self._apply_safety_locked()
                done = controller.done
                success = controller.phase == Phase.SUCCEEDED and not self._safety.failure
                reason = controller.reason
                extra = self._last_collision_status
                status = controller.status
                if not done and extra not in (None, 'READY', 'IDLE', controller.phase.value):
                    status += '/' + extra
            if status != last_status:
                self.get_logger().info(f'Tracking: {status}')
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
                next_tick = time.monotonic()
        return False, 'SHUTDOWN: 추적 노드 종료'

    def execute_callback(self, goal_handle):
        success = False
        message = 'INTERNAL_ERROR: 추적이 정상 종료되지 않음'
        cleanup_ok = False
        try:
            with self._state_lock:
                # AutoNav reaches this action only AFTER NavigateToPose is canceled.
                self._owns_cmd_vel = True
                self._control_finished = False
                self._safety = CollisionSafety(self.collision_config)
                self._probe = Command()
                self._publish_command(force=True)
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
                    ready, readiness_message = self._await_safety_ready(goal_handle)
                    if not ready:
                        message = readiness_message
                    else:
                        success, message = self._run_control(goal_handle)
        except Exception as exc:
            self.get_logger().error(f'Tracking 예외: {exc}')
            message = f'INTERNAL_ERROR: {type(exc).__name__}: {exc}'
            success = False
        finally:
            # Stop BEFORE a potentially slow service call on every exit path.
            with self._state_lock:
                self._control_finished = True
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
            if not cleanup_ok:
                # Preserve both the primary code and success/count semantics.
                # The existing string interface carries cleanup as a separate field.
                message += f'; CLEANUP_UNCONFIRMED: {cleanup_reason}'
                with self._state_lock:
                    self._mode_cleanup_ready = False
            result.success = success
            result.message = message
            with self._state_lock:
                self._publish_command()
                # Release BEFORE publishing action result. Late safe messages
                # and timers can no longer write /cmd_vel into Nav2 patrol.
                self._owns_cmd_vel = False
                self._set_collision_status('IDLE' if success else message.split(':', 1)[0])
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
            with self._state_lock:
                if self._owns_cmd_vel:
                    self._publish_output()
                self._owns_cmd_vel = False
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
        executor = MultiThreadedExecutor(num_threads=4)
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
