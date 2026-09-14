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
from rclpy.qos import qos_profile_sensor_data
from rclpy.clock import Clock, ClockType
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point32, PolygonStamped, Twist
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from navigation_interface.action import RecycleActionMsg

from navigation.tracking_control import (
    Command, Observation, Phase, TrackingConfig, TrackingController,
)

from navigation.collision_safety import CollisionConfig, CollisionSafety, moving


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
        collision_values = {}
        for field in fields(CollisionConfig()):
            name = 'collision_' + field.name
            self.declare_parameter(name, getattr(CollisionConfig(), field.name),
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
        self._safety = CollisionSafety(self.collision_config)
        self._probe = Command()
        self._scan_at = None
        self._scan_stamp_ns = None
        self._scan_frame = ''
        self._scan_error = ''
        self._geometry_at = {'approach': None, 'stop': None}
        self._topology_checked_at = float('-inf')
        self._topology_ok = False
        self._last_collision_status = None
        self._object_msg_seq = 0
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
        self._raw_pub = self.create_publisher(Twist, '/tracking_cmd_vel_raw', 1)
        self._status_pub = self.create_publisher(String, '/tracking_collision/status', 10)
        self._footprint_pub = self.create_publisher(
            PolygonStamped, '/tracking_collision/footprint', 1)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(LaserScan, '/scan', self._scan_callback,
                                 qos_profile_sensor_data, callback_group=self._collision_group)
        self.create_subscription(Twist, '/tracking_cmd_vel_safe', self._safe_callback,
                                 1, callback_group=self._collision_group)
        self.create_subscription(
            PolygonStamped, '/tracking_collision/footprint_checked',
            lambda msg: self._geometry_callback(msg, 'approach'), 1,
            callback_group=self._collision_group)
        self.create_subscription(
            PolygonStamped, '/tracking_collision/hard_stop',
            lambda msg: self._geometry_callback(msg, 'stop'), 1,
            callback_group=self._collision_group)
        # Watchdogs must continue even if ROS simulated time pauses.
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self.collision_config.watchdog_period_sec,
                          self._collision_watchdog, callback_group=self._watchdog_group,
                          clock=self._steady_clock)
        self.create_timer(0.2, self._publish_footprint,
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
            f'stop_only_test_mode={self.config.stop_only_test_mode}'
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
                or self._safety.failure):
            command = Command()
        phase = self._controller.phase.value if self._controller else 'IDLE'
        self._safety.request(command, phase, time.monotonic(), force=force)
        self._raw_pub.publish(self._twist(command))
        if not moving(command):
            self._publish_output()

    def _set_collision_status(self, status):
        if self._last_collision_status != status:
            self._last_collision_status = status
            self._status_pub.publish(String(data=status))
            self.get_logger().info(f'Collision: {status}')

    def _scan_callback(self, msg):
        with self._state_lock:
            now = time.monotonic()
            try:
                stamp_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)
                frame = msg.header.frame_id
                numeric = (msg.angle_min, msg.angle_max, msg.angle_increment,
                           msg.range_min, msg.range_max)
                valid = (bool(frame) and stamp_ns > 0 and len(msg.ranges) >= 2
                         and all(math.isfinite(v) for v in numeric)
                         and msg.angle_increment > 0 and msg.range_max > msg.range_min
                         and any(math.isfinite(v) and max(msg.range_min, 0.001) <= v <= msg.range_max
                                 for v in msg.ranges))
                if not valid:
                    self._scan_error = 'SCAN_INVALID'
                    return
                # A stream repeating the same scan must not renew freshness.
                if self._scan_stamp_ns is not None and stamp_ns <= self._scan_stamp_ns:
                    self._scan_error = 'SCAN_TIMESTAMP_NOT_ADVANCING'
                    return
                self._scan_at = now
                self._scan_stamp_ns = stamp_ns
                self._scan_frame = frame
                self._scan_error = ''
            except (AttributeError, TypeError, ValueError, OverflowError):
                self._scan_error = 'SCAN_INVALID'

    def _geometry_callback(self, msg, kind):
        with self._state_lock:
            try:
                margin = self.collision_config.hard_stop_margin if kind == 'stop' else 0.0
                expected = self.collision_config.rectangle(margin)
                actual = [(float(p.x), float(p.y)) for p in msg.polygon.points]
                # Order/winding may differ, but exactly the same four vertices
                # and base frame must be reported by the active monitor.
                valid = (msg.header.frame_id == 'base_footprint' and len(actual) == 4
                         and all(any(abs(x - ex) < 1e-4 and abs(y - ey) < 1e-4
                                     for x, y in actual) for ex, ey in expected))
                self._geometry_at[kind] = time.monotonic() if valid else None
            except (TypeError, AttributeError, ValueError):
                self._geometry_at[kind] = None

    def _publish_footprint(self):
        msg = PolygonStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.polygon.points = [Point32(x=float(x), y=float(y), z=0.0)
                              for x, y in self.collision_config.rectangle()]
        self._footprint_pub.publish(msg)

    def _environment_ready(self, now):
        cfg = self.collision_config
        if self._scan_error:
            return False, self._scan_error
        if self._scan_at is None:
            return False, 'NO_SCAN'
        if not 0 <= now - self._scan_at <= cfg.scan_timeout_sec:
            return False, 'SCAN_STALE'
        # Receipt freshness alone misses a backlog of old scans. ROS clocks on
        # Pi/PC must agree; time-based gate below fails closed on clock mismatch.
        age = (self.get_clock().now().nanoseconds - self._scan_stamp_ns) * 1e-9
        if age < -cfg.future_stamp_tolerance_sec or age > cfg.scan_timeout_sec:
            return False, 'SCAN_STAMP_STALE_OR_CLOCK_SKEW'
        try:
            if not self._tf_buffer.can_transform('base_footprint', self._scan_frame, Time()):
                return False, 'SCAN_TF_UNAVAILABLE'
        except Exception:
            return False, 'SCAN_TF_UNAVAILABLE'
        for kind, received in self._geometry_at.items():
            if received is None or not 0 <= now - received <= cfg.geometry_timeout_sec:
                return False, 'COLLISION_GEOMETRY_NOT_READY'
        # Avoid connecting a remaining phase-test publisher/monitor to the live
        # production chain by accident. Nodes must be at the expected names.
        if now - self._topology_checked_at > 0.25:
            self._topology_checked_at = now
            try:
                raw = self.get_publishers_info_by_topic('/tracking_cmd_vel_raw')
                safe = self.get_publishers_info_by_topic('/tracking_cmd_vel_safe')
                scans = self.get_publishers_info_by_topic('/scan')
                wheels = self.get_publishers_info_by_topic('/cmd_vel')
                # Existing Nav2/mission publishers remain advertised even after
                # their actions stop. They are governed by the existing cancel
                # handoff, not by publisher count alone. Bench guards/teleop are
                # NOT legitimate simultaneous writers during this test chain.
                allowed_writers = {
                    self.get_name(), 'auto_nav', 'recycle', 'controller_server',
                    'velocity_smoother', 'behavior_server', 'recoveries_server',
                }
                wheel_names = [item.node_name for item in wheels]

                self._topology_ok = (
                    len(raw) == 1 and raw[0].node_name == self.get_name()
                    and len(safe) == 1 and safe[0].node_name == 'tracking_collision_monitor'
                    and len(scans) == 1
                    and all(name in allowed_writers for name in wheel_names)
                    and wheel_names.count(self.get_name()) == 1
                )
            except Exception:
                self._topology_ok = False
        if not self._topology_ok:
            return False, 'COLLISION_TOPIC_OWNERSHIP'
        return True, ''

    def _fail_for_safety(self, reason):
        controller = self._controller
        if controller is not None and controller.phase != Phase.CANCELED:
            # Safety outranks a concurrent visual success/timeout. Otherwise a
            # generic APPROACH_TIMEOUT could make AutoNav restart patrol.
            controller.phase = Phase.FAILED
            controller.reason = f'{reason}: 충돌 안전 경로 중단; 정지 후 운영자 확인 필요'
        self._safety.fail(reason)
        self._publish_output()
        self._set_collision_status(reason)

    def _apply_safety_locked(self, now):
        if not self._owns_cmd_vel:
            return
        controller = self._controller
        if controller is None:
            self._publish_output()
            return
        if self.cancel_event.is_set() or self.shutdown_event.is_set():
            self._publish_output()
            return
        ok, why = self._environment_ready(now)
        decision = self._safety.evaluate(now, ok, why)
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
            try:
                numbers = (msg.linear.x, msg.linear.y, msg.linear.z,
                           msg.angular.x, msg.angular.y, msg.angular.z)
                if (not all(math.isfinite(v) for v in numbers)
                        or any(abs(v) > 1e-9 for v in (
                            msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y))):
                    raise ValueError('Invalid/nonplanar safe Twist')
                self._safety.accept_safe(Command(msg.linear.x, msg.angular.z), time.monotonic())
                self._apply_safety_locked(time.monotonic())
            except Exception as exc:
                self.get_logger().error(f'Collision safe callback: {exc}')
                self._fail_for_safety('SAFETY_INVALID_OUTPUT')

    def _collision_watchdog(self):
        with self._state_lock:
            if not self._owns_cmd_vel:
                return  # No /cmd_vel heartbeats during Nav2 patrol.
            try:
                self._apply_safety_locked(time.monotonic())
            except Exception as exc:
                self.get_logger().error(f'Collision watchdog: {exc}')
                self._fail_for_safety('SAFETY_INTERNAL_ERROR')

    def _await_safety_ready(self, goal_handle):
        deadline = time.monotonic() + self.collision_config.ready_timeout_sec
        reason = 'NO_SAFE'
        while self._running() and time.monotonic() < deadline:
            if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                return False, 'STOP'
            with self._state_lock:
                self._publish_command()
                now = time.monotonic()
                if self._safety.failure:
                    return False, f'{self._safety.failure}: Collision Monitor 응답 오류'
                ok, reason = self._environment_ready(now)
                safe_ready = (self._safety.safe_at is not None
                              and 0 <= now - self._safety.safe_at <= self.collision_config.safe_timeout_sec
                              and self._safety.safe == Command())
                if ok and safe_ready:
                    self._set_collision_status('READY')
                    return True, ''
                self._set_collision_status(reason if not ok else 'NO_SAFE')
            self.cancel_event.wait(self.config.control_period_sec)
        return False, f'SAFETY_NOT_READY: {reason}; LiDAR/TF/Collision Monitor 설정 확인'

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
                now = time.monotonic()
                if self.cancel_event.is_set() or goal_handle.is_cancel_requested:
                    controller.cancel()
                # Evaluate faults/interventions BEFORE the final timer can
                # produce success. The same RLock is used by all callbacks.
                self._apply_safety_locked(now)
                command = controller.step(now)
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
                    self._publish_command()
                else:
                    self._publish_command(command)
                    self._apply_safety_locked(now)
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
                self._safety = CollisionSafety(self.collision_config)
                self._probe = Command()
                self._geometry_at = {'approach': None, 'stop': None}
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
            elif not cleanup_ok and message.split(':', 1)[0] in (
                    'SENSOR_STALE', 'COLLISION_BLOCKED', 'FINAL_APPROACH_INTERRUPTED',
                    'SAFETY_NOT_READY', 'SAFETY_UNAVAILABLE', 'SAFETY_INVALID_OUTPUT',
                    'SAFETY_INVALID_COMMAND', 'SAFETY_INTERNAL_ERROR'):
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
