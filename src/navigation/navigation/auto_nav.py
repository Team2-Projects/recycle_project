import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from action_msgs.msg import GoalStatus
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from std_srvs.srv import Trigger
from rclpy.clock import Clock, ClockType
from dataclasses import fields
from navigation.patrol_recovery import (
    AUTO_PATROL_REASONS, AcquisitionLock, OdomEvidence,
    PatrolRecoveryConfig, ReturnPlan,
)
import json
import time
import math

from my_yolo_msgs.msg import DetectedObject
from navigation_interface.action import RecycleActionMsg
from navigation_interface.srv import ControlServo
from navigation_interface.srv import ControlPantilt

object_name = {0: 'can', 1: 'paper', 2: 'plastic', 3: 'trash', 4: 'person'}

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')

        # --- 추적 실패/센서 홀드 관리 (recycle_tracking.yaml 에서 조정) ---
        self.declare_parameter(
            'tracking_failure_cooldown_sec', 2.0,
            ParameterDescriptor(read_only=True),
        )
        self.tracking_failure_cooldown_sec = float(
            self.get_parameter('tracking_failure_cooldown_sec').value
        )
        if (not math.isfinite(self.tracking_failure_cooldown_sec)
                or self.tracking_failure_cooldown_sec < 0):
            raise ValueError('tracking_failure_cooldown_sec must be finite and nonnegative')
        self._tracking_retry_after = 0.0
        self._tracking_safety_hold = False
        self._tracking_hold_reason = ''
        self.object_found_pub = self.create_publisher(String, "/object_found", 10)
        self.robot_status_pub = self.create_publisher(String, "/robot_status", 10)
        self.robot_task_pub = self.create_publisher(String, "/robot_task", 10)
        self.recycle_success_pub = self.create_publisher(String, "/recycle_success", 10)
        self.schedule_status_pub = self.create_publisher(String, "/schedule_status", 10)

        self.publish_robot_state("state", "Starting")
        self.publish_robot_task("PATROL_PREPARE", "순찰 준비", "", "Task")
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._recycle_client = ActionClient(self, RecycleActionMsg, 'recycle_action')
        self._recycle_tracking_client = ActionClient(self, RecycleActionMsg, 'recycle_tracking_action')
        
        self._action_client.wait_for_server()
        self._recycle_client.wait_for_server()
        # Tracking availability gates collection, not node startup or HOME.
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.servo_client = self.create_client(ControlServo, 'control_servo')

        self.pantilt_client = self.create_client(ControlPantilt, 'control_pantilt')

        self._servo_open = False
        self._servo_epoch = 0
        self.trigger_pantilt_movement(151)
        self.trigger_servo_movement(0, 0, purpose='닫기/시작')

        self.waypoints = []
        self.current_idx = 0

        self.is_running = False
        self.object_found = False
        self.object_id = None
        self.home_x = None
        self.home_y = None

        self.cancel_reason = None
        self.is_returning_home = False

        self.resume_x = None
        self.resume_y = None
        self.current_handle = None
        self.tracking_handle = None
        self.recycle_handle = None

        self.going_home = False  
        self.home_arrive_threshold = 0.5

        self.collected_count = 0         
        self.previous_object_id = None   
        self.y_min = None

        self.abort_retry_count = 0
        self.max_abort_retry = 3

        self.stop_pending = False
        # _servo_open was initialized before the first startup request.
        self.target_x = None
        self.target_y = None
        self.target_h = None
        self.center_x = None
        self.center_y = None

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.create_subscription(Path, '/coverage_path', self.path_callback, latched_qos)
        
        self.object_sub = self.create_subscription(
            DetectedObject,
            '/classified_detected_object_info',
            self.object_callback,
            1
        )

        self.command_sub = self.create_subscription(
            String,
            "/navigation_command",
            self.command_callback,
            10
        )

        self._init_patrol_recovery()

        self.resume_sensor_srv = self.create_service(
            Trigger, '/auto_nav/resume_sensor_hold', self.resume_sensor_hold_callback,
        )
        self.reset_tracking_hold_srv = self.create_service(
            Trigger, '/auto_nav/reset_tracking_hold', self.reset_tracking_hold_callback,
        )
        self.get_logger().info('AutoNav Ready with Multi-collection, Motor, and Web UI integration.')

    def _init_patrol_recovery(self):
        defaults = PatrolRecoveryConfig()
        values = {}
        for field in fields(defaults):
            name = 'patrol_recovery_' + field.name
            self.declare_parameter(name, getattr(defaults, field.name),
                                   ParameterDescriptor(read_only=True))
            values[field.name] = self.get_parameter(name).value
        self._recovery_cfg = PatrolRecoveryConfig(**values)
        # Nav2's configured obstacle layers have no expected_update_rate.
        # Keep a small COMMON scan receipt check, independent of Tracking.
        self.declare_parameter('drive_scan_timeout_sec', .40, ParameterDescriptor(read_only=True))
        self._drive_scan_timeout = float(self.get_parameter('drive_scan_timeout_sec').value)
        if not math.isfinite(self._drive_scan_timeout) or self._drive_scan_timeout <= 0:
            raise ValueError('drive_scan_timeout_sec must be finite and positive')
        self._drive_scan = None
        self._odom = OdomEvidence(self._recovery_cfg)
        self._acquisition_lock = AcquisitionLock(self._recovery_cfg.release_distance_m,
                                                  self._recovery_cfg.odom_timeout_sec)
        self._return_plan = None
        # Stop/recovery invalidates old post-collection timer callbacks. Each
        # tracking request/result also has a local generation, not just a bool.
        self._collection_generation = 0
        self._tracking_request_generation = 0
        self._tracking_response_generation = None
        self._tracking_result_generation = None
        # Our adapter releases wheel ownership BEFORE returning a terminal
        # result. No independent Tracking health subscription is needed.
        self._tracking_release_confirmed = True
        self._nav_request_generation = 0
        self._nav_response_generation = None
        self._nav_result_generation = None
        self._nav_goal_purpose = 'PATROL'
        self._recycle_request_generation = 0
        self._recycle_response_generation = None
        self._recycle_result_generation = None
        self._home_completed = False
        self.check_timer = None
        self.delay_timer = None
        self._nav_goal_pending = False
        self._tracking_goal_pending = False
        self._recycle_goal_pending = False
        self._close_future = None
        self._close_started_at = None
        self._close_attempts = 0
        self._close_retry_after = 0.0
        self._last_recovery_status = None
        self._recovery_status_since = time.monotonic()
        self._recovery_status_at = -math.inf
        self._recovery_status_sequence = 0
        self._recovery_pub = self.create_publisher(String, '/auto_nav/recovery_status', 10)
        self._recovery_details_pub = self.create_publisher(String, '/auto_nav/recovery_details', 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Odometry, '/odom', self._odom_callback, qos)
        self.create_subscription(LaserScan, '/scan', self._drive_scan_callback, qos)
        self._recovery_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self._recovery_cfg.period_sec, self._auto_return_tick,
                          clock=self._recovery_clock)
        self.get_logger().info('patrol_recovery_revision=v3; 보완접근 중단도 대상 포기 후 조건부 자동 순찰 복귀')

    def _set_recovery_status(self, state):
        now = time.monotonic()
        if state != self._last_recovery_status:
            self._last_recovery_status = state
            self._recovery_status_since = now
            self._recovery_status_at = -math.inf
            self.get_logger().info(f'Patrol recovery: {state}')
        if now - self._recovery_status_at < 1.0:
            return
        self._recovery_status_at = now
        self._recovery_status_sequence += 1
        plan = self._return_plan
        details = dict(
            state=state, sequence=self._recovery_status_sequence,
            destination=(plan.destination if plan else 'HOME' if self.is_returning_home else None),
            reason=plan.reason if plan else self._tracking_hold_reason,
            elapsed_sec=round(max(0.0, now - plan.created_at), 1) if plan else 0.0,
            state_elapsed_sec=round(max(0.0, now - self._recovery_status_since), 1),
            manual=plan.manual if plan else False,
            waiting=state.startswith('RECOVERY_WAIT_') or state in ('HOME_WAIT_GOAL', 'MANUAL_HOLD'),
        )
        # Preserve the existing string topic; structured details are telemetry,
        # never another motion gate or a timeout-based drive authorization.
        self._recovery_pub.publish(String(data=state))
        self._recovery_details_pub.publish(String(data=json.dumps(details)))

    def _drive_scan_callback(self, msg):
        now, ros_ns = time.monotonic(), self.get_clock().now().nanoseconds
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        if (not msg.header.frame_id or not 0 <= msg.header.stamp.nanosec < 10**9
                or stamp <= 0 or len(msg.ranges) < 2
                or not all(math.isfinite(v) for v in (msg.angle_min, msg.angle_max,
                            msg.angle_increment, msg.range_min, msg.range_max))
                or msg.angle_increment <= 0 or msg.range_max <= msg.range_min
                or not any(math.isfinite(v) and max(msg.range_min, .001) <= v <= msg.range_max
                           for v in msg.ranges)
                or not -self._recovery_cfg.future_stamp_tolerance_sec
                <= (ros_ns - stamp) * 1e-9 <= self._drive_scan_timeout):
            return
        if self._drive_scan is not None:
            previous, received = self._drive_scan
            if stamp == previous or (stamp < previous and now - received <= self._drive_scan_timeout):
                return
        self._drive_scan = stamp, now

    def _drive_scan_fresh(self, now):
        return (self._drive_scan is not None
                and 0 <= now - self._drive_scan[1] <= self._drive_scan_timeout
                and -self._recovery_cfg.future_stamp_tolerance_sec
                <= (self.get_clock().now().nanoseconds - self._drive_scan[0]) * 1e-9
                <= self._drive_scan_timeout)

    def _odom_callback(self, msg):
        try:
            now = time.monotonic()
            p, v = msg.pose.pose.position, msg.twist.twist
            stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            if (not 0 <= int(msg.header.stamp.nanosec) < 10**9
                    or msg.child_frame_id not in ('base_link', 'base_footprint')):
                return
            accepted = self._odom.receive(
                now, self.get_clock().now().nanoseconds, stamp_ns, msg.header.frame_id,
                float(p.x), float(p.y), math.hypot(v.linear.x, v.linear.y), float(v.angular.z))
            if accepted:
                self._acquisition_lock.observe(self._odom.x, self._odom.y, now)
        except (ValueError, TypeError, AttributeError, OverflowError):
            pass

    def _any_goal_pending(self):
        return self._nav_goal_pending or self._tracking_goal_pending or self._recycle_goal_pending

    def _valid_resume_goal(self):
        return (self.is_running and all(type(v) in (int, float) and math.isfinite(v)
                                       for v in (self.resume_x, self.resume_y)))

    def _cancel_collection_timers(self):
        """Keep delayed success callbacks from restarting patrol after STOP/failure."""
        self._collection_generation += 1
        for name in ('check_timer', 'delay_timer'):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
                setattr(self, name, None)

    def _collection_continuation_allowed(self, generation):
        return (generation == self._collection_generation and self.is_running
                and self.cancel_reason not in ('STOP', 'BATTERY_LOW')
                and not self.stop_pending and not self.is_returning_home
                and not self._tracking_safety_hold and self._return_plan is None
                and not self._any_goal_pending() and self.current_handle is None
                and self.tracking_handle is None and self.recycle_handle is None)

    def _cancel_auto_return(self, state):
        self._return_plan = None
        if self._close_future is not None and not self._close_future.done():
            # Cancels the local waiter only, NOT server-side servo movement.
            self._close_future.cancel()
        self._close_future = None
        self._set_recovery_status(state)

    def _queue_auto_return(self, reason, manual=False, destination='PATROL'):
        self._cancel_auto_return('RECOVERY_QUEUED')
        x, y = ((self.home_x, self.home_y) if destination == 'HOME'
                else (self.resume_x, self.resume_y))
        self._return_plan = ReturnPlan(reason, float(x), float(y), time.monotonic(),
                                       manual=manual, destination=destination)
        if destination == 'PATROL':
            self._acquisition_lock.engage()
        self._tracking_safety_hold = True
        self.object_found = True
        self._close_attempts = 0
        self._close_retry_after = 0.0
        label = ('보완접근 중단·이번 수거 미확정·순찰 복귀 준비'
                 if reason == 'FINAL_APPROACH_INTERRUPTED' else '수거 대상 포기·순찰 복귀 준비')
        self.publish_robot_task('HOME_PREPARE' if destination == 'HOME' else 'TRACKING_RECOVERY',
                                'HOME 복귀 준비' if destination == 'HOME' else label, reason, 'Task')

    def _poll_recovery_close(self, now):
        plan = self._return_plan
        if plan.servo_confirmed:
            return True
        if self._close_future is not None:
            if self._close_future.done():
                try:
                    result = self._close_future.result()
                    good = result is not None and getattr(result, 'success', False) is True
                except Exception:
                    good = False
                self._close_future = None
                if good:
                    plan.servo_confirmed = True
                    self._servo_open = False
                    return True
                self.get_logger().warn('순찰 복귀용 서보 닫힘 응답 실패; 재확인 대기')
            elif now - self._close_started_at <= self._recovery_cfg.service_timeout_sec:
                return False
            else:
                self._close_future.cancel()
                self._close_future = None
                self.get_logger().warn('순찰 복귀용 서보 응답 시간 초과; 정지 상태에서 재확인')
            # Initial request + one quick retry; later batches are spaced out.
            if self._close_attempts >= 2:
                self._close_attempts = 0
                self._close_retry_after = now + self._recovery_cfg.service_retry_sec
        if now < self._close_retry_after or not self.servo_client.service_is_ready():
            return False
        request = ControlServo.Request()
        request.angle1 = request.angle2 = 0.0
        self._servo_epoch += 1  # invalidate late callbacks from the earlier open
        try:
            self._close_future = self.servo_client.call_async(request)
            self._close_started_at = now
            self._close_attempts += 1
        except Exception as exc:
            self._close_retry_after = now + self._recovery_cfg.service_retry_sec
            self.get_logger().warn(f'복귀 서보 서비스 요청 실패: {exc}')
        return False

    def _auto_return_tick(self):
        now = time.monotonic()
        plan = self._return_plan
        if plan is None:
            self._set_recovery_status(self._last_recovery_status or 'IDLE')
            return
        if plan.destination == 'PATROL':
            if self.is_returning_home or self.cancel_reason in ('STOP', 'BATTERY_LOW'):
                self.return_home_by_stop()
                return
            if not self.is_running:
                self._cancel_auto_return('RECOVERY_CANCELED_BY_OPERATOR')
                return
        if (self._any_goal_pending() or self.current_handle is not None
                or self.tracking_handle is not None or self.recycle_handle is not None):
            self._set_recovery_status('RECOVERY_WAIT_ACTION_FINISH')
            return  # Never fight an active Nav2/Tracking goal with zero heartbeats.
        self.stop_pending = False
        self.cmd_vel_pub.publish(Twist())
        # A terminal result is not proof of physical stopping. Do not close
        # while tracking still owns motion or odometry still reports motion.
        if not self._tracking_release_confirmed:
            self._set_recovery_status('RECOVERY_WAIT_TRACKING_RELEASE')
            return
        ros_ns = self.get_clock().now().nanoseconds
        if not self._odom.stopped(now, ros_ns):
            self._set_recovery_status('RECOVERY_WAIT_STOP' if self._odom.fresh(now, ros_ns)
                                      else 'RECOVERY_WAIT_ODOM')
            return
        if not self._poll_recovery_close(now):
            self._set_recovery_status('RECOVERY_WAIT_SERVO')
            return
        if not self._drive_scan_fresh(now):
            self._set_recovery_status('RECOVERY_WAIT_SCAN_STALE')
            return
        # Nav2 owns localization, costmaps and controller lifecycle. Readiness
        # is its Action server contract, not a second lifecycle health FSM.
        if not self._action_client.server_is_ready():
            self._set_recovery_status('RECOVERY_WAIT_NAV2_SERVER')
            return
        # Single-threaded AutoNav callbacks: no asynchronous wait between this
        # final check and send_goal(). Pending-goal flag is set before sending.
        self._return_plan = None
        if plan.destination == 'HOME':
            self._set_recovery_status('HOME_STARTING')
            self.publish_robot_state('state', 'Return Home')
            self.send_goal(plan.x, plan.y, purpose='HOME')
            return
        self._tracking_safety_hold = False
        self._tracking_hold_reason = ''
        self.object_found = False
        if self.collected_count == 0:
            self.object_id = None
            self.previous_object_id = None
        if self.cancel_reason == 'OBJECT':
            self.cancel_reason = None
        self._tracking_retry_after = now + self.tracking_failure_cooldown_sec
        self._acquisition_lock.begin_patrol(self._odom.x, self._odom.y, now)
        self._set_recovery_status('PATROL_RESUMING')
        self.publish_robot_task('PATROL_RESUME', '순찰 자동 복귀',
                                 f'{plan.reason}; 현재 대상 포기, 다음 실제 waypoint까지 수거 잠금', 'Task')
        try:
            self.send_goal(plan.x, plan.y)
        except Exception as exc:
            # Unknown goal submission errors are not evidence it was rejected;
            # do not send a second concurrent navigation action automatically.
            self._handle_tracking_failure(f'INTERNAL_ERROR: 순찰 복귀 요청 오류: {exc}')

    def trigger_servo_movement(self, angle1, angle2, purpose='', verify=False, retries=1):
        """서보 각도 요청. verify=True 면 ControlServo 응답까지 확인한다.

        기존 코드는 call_async 결과를 보지 않아 서보가 실제로 움직였는지 알 수
        없었다. 막이가 열린 채로 순찰을 계속하면 다음 대상 접근이나 주행에
        지장이 있으므로, 열기/닫기는 응답을 확인하고 실패 시 1회 재시도한다.
        """
        label = purpose or f'angle=({angle1}, {angle2})'
        req = ControlServo.Request()
        req.angle1 = float(angle1)
        req.angle2 = float(angle2)
        if not self.servo_client.service_is_ready():
            self.get_logger().warn(f'서보 서비스 미준비 상태에서 요청 ({label})')
        self._servo_epoch += 1
        epoch = self._servo_epoch
        self.servo_future = self.servo_client.call_async(req)
        if verify:
            self.servo_future.add_done_callback(
                lambda fut: self._servo_result_callback(fut, angle1, angle2, label, retries, epoch)
            )
        else:
            # 확인하지 않는 호출도 목표 상태는 기록해 둔다.
            self._servo_open = (float(angle1), float(angle2)) != (0.0, 0.0)
        return self.servo_future

    def _servo_result_callback(self, future, angle1, angle2, label, retries, epoch=None):
        """ControlServo 응답 확인. 실패하면 재시도하고, 그래도 실패하면 알린다."""
        # A late open/retry response must not undo a newer recovery close.
        if epoch is not None and epoch != self._servo_epoch:
            return
        detail = ''
        try:
            response = future.result()
        except Exception as exc:
            response = None
            detail = f'응답 수신 오류: {exc}'
        if response is not None:
            if getattr(response, 'success', False):
                self._servo_open = (float(angle1), float(angle2)) != (0.0, 0.0)
                state = '열림' if self._servo_open else '닫힘'
                self.get_logger().info(f'서보 {state} 확인 ({label})')
                return
            detail = getattr(response, 'message', '') or '서비스가 success=False 반환'

        if retries > 0:
            self.get_logger().warn(f'서보 동작 실패 ({label}): {detail} → 재시도')
            self.trigger_servo_movement(
                angle1, angle2, purpose=label, verify=True, retries=retries - 1
            )
            return

        # 실제 상태를 알 수 없으므로 열린 것으로 간주해 이후 닫기 시도를 허용한다.
        self._servo_open = True
        self.get_logger().error(f'서보 동작 확인 실패 ({label}): {detail}')
        self.publish_robot_task(
            'SERVO_FAIL', '서보 동작 확인 실패', f'{label}: {detail}', 'Error'
        )

    def close_servo_if_open(self, reason):
        """열린 막이를 중립(0, 0)으로 되돌린다. 이미 닫혀 있으면 아무것도 하지 않는다."""
        if not self._servo_open:
            return
        self.get_logger().info(f'막이 닫기 요청 ({reason})')
        self.trigger_servo_movement(0, 0, purpose=f'닫기/{reason}', verify=True)

    def trigger_pantilt_movement(self, angle):
        req = ControlPantilt.Request()
        req.angle = float(angle)
        self.pantilt_future = self.pantilt_client.call_async(req)

    def publish_recycle_success(self, obj_name, confidence):
        msg = String()
        msg.data = json.dumps({
            "object_name": obj_name,
            "confidence": confidence,
            "status": "Success"
        })
        self.recycle_success_pub.publish(msg)

    def publish_object_found(self, obj_name, confidence):
        msg = String()
        msg.data = json.dumps({
            "object_name": obj_name,
            "confidence": confidence,
            "status": "Success"
        })
        self.object_found_pub.publish(msg)

    def publish_robot_state(self, eventType, status):
        msg = String()
        msg.data = json.dumps({
            "eventType": eventType,
            "status": status
        })
        self.robot_status_pub.publish(msg)

    def publish_robot_task(self, eventType, message, note, status):
        msg = String()
        msg.data = json.dumps({
            "eventType": eventType,
            "message": message,
            "note": note,
            "status": status
        })
        self.robot_task_pub.publish(msg)

    def publish_schedule_status(self, status):
        msg = String()
        msg.data = json.dumps({
            "status": status
        })
        self.schedule_status_pub.publish(msg)

    def command_callback(self, msg):
        if msg.data == "STOP":
            self.cancel_reason = "STOP"
        elif msg.data == "BATTERY_LOW":
            self.cancel_reason = "BATTERY_LOW"
        else:
            return

        self._cancel_collection_timers()
        self.publish_schedule_status("CANCEL")
        self.return_home_by_stop()
        self.stop_pending = self._any_goal_pending()

        if self.tracking_handle is not None:
            self.get_logger().info('Tracking Action 취소 요청')
            self.tracking_handle.cancel_goal_async()

        if self.recycle_handle is not None:
            self.get_logger().info('Recycle Action 취소 요청')
            self.recycle_handle.cancel_goal_async()

        if self.current_handle is not None and self._nav_goal_purpose != 'HOME':
            self.get_logger().info('NavigateToPose 취소 요청')
            self.current_handle.cancel_goal_async()

    def return_home_by_stop(self):
        if self._home_completed:
            return
        if (self._return_plan is not None and self._return_plan.destination == 'HOME'
                or self._nav_goal_purpose == 'HOME'
                and (self._nav_goal_pending or self.current_handle is not None)):
            return
        self._cancel_collection_timers()
        if not self.is_returning_home:
            self.abort_retry_count = 0
        self.is_returning_home = True
        self.object_found = True
        if not all(type(v) in (int, float) and math.isfinite(v)
                   for v in (self.home_x, self.home_y)):
            self._cancel_auto_return('HOME_WAIT_GOAL')
            return  # path_callback supplies HOME when the first path arrives.
        if self.cancel_reason == "STOP":
            self.publish_robot_task('USER_COMMAND', '사용자 명령', '순찰 종료', 'Task')
        elif self.cancel_reason == "BATTERY_LOW":
            self.publish_robot_task('BATTERY_LOW', '배터리 경고', '배터리가 30% 이하', 'WARNING')

        self.publish_object_found("-", "-")
        self.publish_robot_state('state', 'Return Home')

        self._queue_auto_return(self.cancel_reason or 'STOP', destination='HOME')

    def path_callback(self, msg):
        if self.is_running:
            self.get_logger().warn('Already navigating, ignoring new path')
            return

        if not msg.poses:
            self.get_logger().error('수신된 경로에 waypoint가 없습니다.')
            return

        self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.home_x = self.waypoints[-1][0]
        self.home_y = self.waypoints[-1][1]

        self.current_idx = 0
        self.is_running = True
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        if self.is_returning_home:
            self.return_home_by_stop()
            return
        self.send_next_goal()

        self.publish_robot_state("state", "Running")
        self.publish_robot_task("PATROL_START", "순찰 시작", "", "Task")

    def resume_sensor_hold_callback(self, request, response):
        """Explicit retry; Tracking alone verifies fresh vision, Monitor and OFF completion."""
        response.success = False
        retryable_reasons = ('SENSOR_STALE', 'VISION_NOT_READY')
        if not self._tracking_safety_hold or self._tracking_hold_reason not in retryable_reasons:
            response.message = (
                'SENSOR_STALE/VISION_NOT_READY 정지에서만 동일 대상 재시도 가능; '
                '기타 홀드는 /auto_nav/reset_tracking_hold 사용'
            )
            return response
        if (self.cancel_reason in ('STOP', 'BATTERY_LOW') or self.is_returning_home
                or self.stop_pending or self._any_goal_pending()
                or not self.is_running or self.tracking_handle is not None
                or self.current_handle is not None or self.recycle_handle is not None):
            response.message = '다른 작업/취소/HOME 복귀 상태이므로 재개하지 않음'
            return response
        if self.object_id is None:
            response.message = '재시도할 대상 없음'
            return response
        if not self._recycle_tracking_client.server_is_ready():
            response.message = '추적 Action 서버가 준비되지 않음'
            return response
        self._cancel_auto_return('MANUAL_RETRY')
        self._acquisition_lock = AcquisitionLock(self._recovery_cfg.release_distance_m,
                                                  self._recovery_cfg.odom_timeout_sec)
        self.cmd_vel_pub.publish(Twist())
        # 실패 시 닫았으므로 재시도 전에 다시 연다.
        self.trigger_servo_movement(-90, 90, purpose='열기/추적재시도', verify=True)
        self.cancel_reason = 'OBJECT'
        self._tracking_safety_hold = False
        self._tracking_hold_reason = ''
        self.object_found = True
        try:
            self.launch_recycle_tracking_action()
        except Exception as exc:
            self._handle_tracking_failure(f'INTERNAL_ERROR: 복구 요청 실패: {exc}')
            response.message = str(exc)
            return response
        response.success = True
        response.message = '동일 대상 재정렬 작업 요청 완료; 실제 진행은 Action 결과로 확인'
        return response

    def reset_tracking_hold_callback(self, request, response):
        """Operator explicitly abandons a hold, but never bypasses live inputs."""
        response.success = False
        if not self._tracking_safety_hold:
            response.message = '해제할 tracking hold가 없음'
            return response
        if (self.cancel_reason in ('STOP', 'BATTERY_LOW') or self.stop_pending
                or self.is_returning_home or self._any_goal_pending()
                or self.tracking_handle is not None or self.current_handle is not None
                or self.recycle_handle is not None):
            response.message = '다른 작업/사용자 STOP/HOME 상태이므로 재개하지 않음'
            return response
        if not self._valid_resume_goal():
            response.message = '재개할 순찰 목표가 없음; 정지 유지'
            return response
        if self._return_plan is None:
            self._queue_auto_return(self._tracking_hold_reason or 'MANUAL_RESET', manual=True)
        response.success = True
        response.message = ('대상 포기 요청 등록; 공통 scan/정지 odom과 서보 닫힘을 확인한 뒤 '
                            '자동 순찰 재개. /auto_nav/recovery_status 확인')
        return response

    def _handle_tracking_failure(self, message):
        """Abort ONE collection, not the mission. No blind reset-and-drive."""
        self._cancel_collection_timers()
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().warn(f'Tracking 종료: {message}')
        if self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.return_home_by_stop()
            return
        code = message.split(':', 1)[0].strip()
        self._tracking_hold_reason = code
        self._tracking_retry_after = time.monotonic() + self.tracking_failure_cooldown_sec
        self.publish_robot_task('OBJECT_PICKUP_FAIL', '추적 종료', message, 'Warning')
        self._tracking_safety_hold = True
        self.object_found = True
        final_return_allowed = (
            code != 'FINAL_APPROACH_INTERRUPTED' or self._recovery_cfg.final_interrupted_enabled)
        if (self._recovery_cfg.enabled and code in AUTO_PATROL_REASONS
                and final_return_allowed
                and self._valid_resume_goal() and not self.stop_pending):
            self._queue_auto_return(code)
        else:
            self._cancel_auto_return('MANUAL_HOLD')
            self.close_servo_if_open('추적 실패')
            self.publish_robot_task('TRACKING_HOLD', '정지 확인 필요', message, 'Warning')

    def object_callback(self, msg):
        if msg.id == -1:
            return

        # 좌표/신뢰도가 깨진 결과로 추적을 시작하지 않는다.
        try:
            values = [float(v) for v in msg.coord]
            conf = float(msg.confidence)
            if (msg.id < 0 or len(values) != 4 or not 0.0 < conf <= 1.0
                    or not all(math.isfinite(v) for v in values)
                    or values[0] < 0 or values[1] < 0
                    or values[2] <= 0 or values[3] <= 0):
                return
        except (TypeError, ValueError, AttributeError, OverflowError):
            return

        try:
            value = float(getattr(msg, 'min_y', 0))
            self.y_min = value if math.isfinite(value) else None
        except (TypeError, ValueError):
            self.y_min = None
        if self.collected_count > 0 and msg.id != self.previous_object_id:
            return 
        
        if self.object_found:
            return

        # 정지 확인이 필요한 홀드 상태이거나 실패 직후 쿨다운 중이면 재시작하지 않는다.
        if (self._tracking_safety_hold or self._acquisition_lock.locked
                or self._return_plan is not None or self._any_goal_pending()
                or time.monotonic() < self._tracking_retry_after
                or self.cancel_reason in ('STOP', 'BATTERY_LOW')
                or self.is_returning_home or not self.is_running
                or self.current_handle is None):
            return


        self.target_x, self.target_y, self.target_h = values[0], values[1], values[3]
        self.object_id = msg.id
        obj_name = object_name.get(msg.id, '-')
        conf_val = f"{msg.confidence:.2f}" if hasattr(msg, 'confidence') else "1.00"
        self.publish_recycle_success(obj_name, conf_val)

        if self.collected_count == 0:
            self.previous_object_id = msg.id
            self.get_logger().info(f'🎯 Target Object ID set to: {self.previous_object_id}')

        self.object_found = True

        self.get_logger().info("Object detected! Triggering servo...")
        self.trigger_servo_movement(-90, 90, purpose='열기/객체감지', verify=True)

        obj_str = object_name.get(msg.id, '-')
        conf_val = f"{msg.confidence:.2f}" if hasattr(msg, 'confidence') else "1.00"
        
        self.publish_object_found(obj_str, conf_val)
        self.publish_robot_task(
            "OBJECT_DETECTED",
            "물체 감지",
            f"물체: {obj_str} / 신뢰도: {conf_val}",
            "Detect"
        )


        if self.current_idx < len(self.waypoints):
            self.resume_x, self.resume_y = self.waypoints[self.current_idx]
        
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        if self.current_handle is not None:
            self.cancel_reason = "OBJECT"
            self.current_handle.cancel_goal_async()

    def launch_recycle_tracking_action(self):
        if self._tracking_goal_pending or self.tracking_handle is not None:
            return
        if self.is_returning_home or self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.return_home_by_stop()
            return
        if not self._recycle_tracking_client.server_is_ready():
            self._handle_tracking_failure('TRACKING_GOAL_REJECTED: 추적 Action 서버 미준비; 대상 포기')
            return
        self.publish_robot_task("OBJECT_PICKUP_START", "수거 시작", "", "Task")
        
        goal_msg = RecycleActionMsg.Goal()
        # index = 추적할 클래스 ID. SetTracking.target_class_id 로 전달되어
        # 인식 노드가 다른 클래스로 대상을 바꾸지 않게 잠근다.
        goal_msg.index = int(self.object_id if self.object_id is not None else 0)
        goal_msg.target_x = float(self.target_x)
        goal_msg.target_y = float(self.target_y)
        goal_msg.target_h = float(self.target_h)
        
        self.get_logger().info('🚀 recycle_tracking_action 호출 (회전 + 접근)')
        self._tracking_request_generation += 1
        generation = self._tracking_request_generation
        self._tracking_goal_pending = True
        self._tracking_release_confirmed = False
        try:
            future = self._recycle_tracking_client.send_goal_async(goal_msg)
        except Exception:
            self._tracking_goal_pending = False
            raise
        future.add_done_callback(
            lambda fut: self.recycle_tracking_goal_response_callback(fut, generation))

    def recycle_tracking_goal_response_callback(self, future, generation=None):
        if generation is not None and generation == self._tracking_response_generation:
            return
        if generation is not None and generation != self._tracking_request_generation:
            # A delayed old acceptance must not become the current handle.
            try:
                old_handle = future.result()
                if old_handle is not None and old_handle.accepted:
                    old_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warn(f'과거 Tracking 응답 처리: {exc}')
            return
        self._tracking_response_generation = generation
        self._tracking_goal_pending = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._handle_tracking_failure(f'INTERNAL_ERROR: 추적 요청 응답 오류: {exc}')
            return
        if not goal_handle.accepted:
            self._tracking_release_confirmed = True
            self._handle_tracking_failure('TRACKING_GOAL_REJECTED: 추적 요청이 거절됨; 대상 포기')
            return

        self.tracking_handle = goal_handle

        if self.stop_pending or self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.stop_pending = False
            self.tracking_handle.cancel_goal_async()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut: self.recycle_tracking_result_callback(fut, generation))

    def recycle_tracking_result_callback(self, future, generation=None):
        if generation is not None:
            if (generation != self._tracking_request_generation
                    or generation == self._tracking_result_generation):
                return  # Late/duplicate results cannot count or restart a job.
            self._tracking_result_generation = generation
        self.tracking_handle = None
        try:
            response = future.result()
            status = response.status
            result = response.result
        except Exception as exc:
            self._handle_tracking_failure(f'INTERNAL_ERROR: 추적 결과 수신 오류: {exc}')
            return

        if status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
            self._tracking_release_confirmed = True
        if self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.cmd_vel_pub.publish(Twist())
            self.return_home_by_stop()
            return

        if status == GoalStatus.STATUS_CANCELED or result.message == 'STOP':
            self._handle_tracking_failure('TRACKING_CANCELED: 추적 Action 취소됨')
            return

        if not result.success or status != GoalStatus.STATUS_SUCCEEDED:
            self._handle_tracking_failure(result.message or 'INTERNAL_ERROR: 빈 실패 응답')
            return

        # Only an actual SUCCEEDED result increments collection; interrupted
        # final movement uses the failure/return path above, never this pipeline.
        self._cancel_collection_timers()
        self.get_logger().info("Successed tracking! Triggering servo & pantilt...")
        self.trigger_servo_movement(0, 0, purpose='닫기/수거성공', verify=True)
        self.trigger_pantilt_movement(90)

        self.collected_count += 1
        self.get_logger().info(f'📦 물품 수거 성공! (현재 수거량: {self.collected_count})')
        
        self.y_min = None
        self.get_logger().info('⏳ 3초간 수거함 상태 확인 중...')
        generation = self._collection_generation
        self.check_timer = self.create_timer(
            3.0, lambda: self.check_recycle_condition_callback(generation))

    def _delayed_resume(self, generation=None):
        generation = self._collection_generation if generation is None else generation
        if generation != self._collection_generation:
            return
        if self.delay_timer is not None:
            self.delay_timer.cancel()
            self.destroy_timer(self.delay_timer)
            self.delay_timer = None
        if not self._collection_continuation_allowed(generation):
            return
        self.object_found = False
        self.send_goal(self.resume_x, self.resume_y)

    def check_recycle_condition_callback(self, generation=None):
        generation = self._collection_generation if generation is None else generation
        if generation != self._collection_generation:
            return
        if self.check_timer is not None:
            self.check_timer.cancel()
            self.destroy_timer(self.check_timer)
            self.check_timer = None
        if not self._collection_continuation_allowed(generation):
            return

        if self.y_min is not None and 0 <= self.y_min <= 180:
            self.object_found = True  
            self.get_logger().info(f'🗑️ 수거함 포화 감지 (y_min: {self.y_min:.1f})! HOME으로 이동합니다.')
            self.trigger_pantilt_movement(151)
            self.launch_recycle_action()
        else:
            self.trigger_pantilt_movement(151)
            self.get_logger().info('🔄 수거 완료. 순찰을 계속합니다.')
            if self.object_found:
                self.delay_timer = self.create_timer(
                    2.0, lambda: self._delayed_resume(generation))

    def launch_recycle_action(self):
        if self._recycle_goal_pending or self.recycle_handle is not None:
            return
        if self.is_returning_home or self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.return_home_by_stop()
            return
        goal_msg = RecycleActionMsg.Goal()
        goal_msg.index = self.object_id if self.object_id is not None else 1
        goal_msg.current_idx = self.current_idx
        goal_msg.home_x = self.home_x
        goal_msg.home_y = self.home_y

        self.get_logger().info('🚀 recycle_action 호출 (HOME 이동 + 후진 + 회전)')
        self._recycle_goal_pending = True
        self._recycle_request_generation += 1
        generation = self._recycle_request_generation
        try:
            future = self._recycle_client.send_goal_async(goal_msg)
        except Exception:
            self._recycle_goal_pending = False
            raise
        future.add_done_callback(lambda fut: self.recycle_goal_response_callback(fut, generation))

    def recycle_goal_response_callback(self, future, generation=None):
        if generation is not None and (generation != self._recycle_request_generation
                                       or generation == self._recycle_response_generation):
            return
        self._recycle_response_generation = generation
        self._recycle_goal_pending = False
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self.is_returning_home:
                self.return_home_by_stop()
                return
            self.get_logger().error('❌ recycle 목표 거절됨')
            self.object_found = False
            return

        self.recycle_handle = goal_handle

        if self.stop_pending or self.is_returning_home:
            self.stop_pending = False
            self.recycle_handle.cancel_goal_async()
            
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self.recycle_result_callback(fut, generation))

    def recycle_result_callback(self, future, generation=None):
        if generation is not None and (generation != self._recycle_request_generation
                                       or generation == self._recycle_result_generation):
            return
        self._recycle_result_generation = generation
        response = future.result()
        status = response.status
        result = response.result
        self.recycle_handle = None

        if self.is_returning_home or self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.return_home_by_stop()
            return
        if status == GoalStatus.STATUS_CANCELED:
            self.return_home_by_stop()
            return

        if not result.success:
            self.get_logger().warn(
                f'분리수거장 이동 실패: {result.message}'
            )
            self.publish_robot_task(
                'OBJECT_PICKUP_FAIL',
                '분리수거 실패',
                '',
                'Error'
            )
            if rclpy.ok():
                rclpy.shutdown()
            return

        self.collected_count = 0
        self.previous_object_id = None
        self.object_found = False

        self.publish_robot_task(
            "PATROL_RESUME",
            "순찰 재개",
            "",
            "Task"
        )

        self.get_logger().info(
            '↩️ 버리기 완료! 원래 목표로 복귀 시작'
        )

        self.current_idx = 0
        self.send_next_goal()

    def send_next_goal(self):
        if self.is_returning_home or self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.return_home_by_stop()
            return
        if self.current_idx >= len(self.waypoints):
            if self.collected_count > 0:
                self.publish_robot_task("OBJECT_PICKUP_START", "수거 시작", "", "Task")
                self.object_found = True
                self.launch_recycle_action()
                return 
            
            self.publish_robot_state("state", "Stop")
            self.publish_robot_task("PATROL_COMPLETE", "순찰 종료", "", "Task")
            self.publish_schedule_status("COMPLETE")
            self.get_logger().info('🏁 Patrol finished. Shutting down...')

            if rclpy.ok():
                rclpy.shutdown()
            return

        x, y = self.waypoints[self.current_idx]
        # The final waypoint is HOME; finish this leg without a new collection.
        if self.current_idx > len(self.waypoints) - 2:
            self.object_found = True
        self.send_goal(x, y)

    def send_goal(self, x, y, purpose='PATROL'):
        if purpose != 'HOME' and (self.is_returning_home
                                  or self.cancel_reason in ('STOP', 'BATTERY_LOW')):
            self.return_home_by_stop()
            return
        if self._nav_goal_pending or self.current_handle is not None:
            self.get_logger().warn('기존 Nav2 요청/목표 처리 중; 중복 목표 전송하지 않음')
            return
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self._nav_goal_pending = True
        self._nav_request_generation += 1
        generation = self._nav_request_generation
        self._nav_goal_purpose = purpose
        try:
            future = self._action_client.send_goal_async(
                goal_msg, feedback_callback=lambda msg: self.feedback_callback(msg, generation))
        except Exception:
            self._nav_goal_pending = False
            raise
        future.add_done_callback(lambda fut: self.goal_response_callback(fut, generation, purpose))

    def goal_response_callback(self, future, generation=None, purpose=None):
        if generation is not None and (generation != self._nav_request_generation
                                       or generation == self._nav_response_generation):
            return
        self._nav_response_generation = generation
        purpose = purpose or self._nav_goal_purpose
        self._nav_goal_pending = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._handle_tracking_failure(f'INTERNAL_ERROR: Nav2 요청 응답 오류: {exc}')
            return
        if not goal_handle.accepted and purpose != 'HOME' and self.is_returning_home:
            self.return_home_by_stop()
            return
        if not goal_handle.accepted:
            if purpose == 'HOME':
                self.get_logger().error('HOME Goal rejected!')
                self.publish_robot_state("state", "Error")
                return

            self.get_logger().warn('Goal rejected! Skipping.')
            self.current_idx += 1
            self.send_next_goal()
            return

        self.current_handle = goal_handle

        if purpose != 'HOME' and (self.stop_pending or self.is_returning_home):
            self.stop_pending = False

            self.get_logger().info(
                '대기 중이던 STOP 요청 처리 → NavigateToPose 취소'
            )

            self.current_handle.cancel_goal_async()
        goal_handle.get_result_async().add_done_callback(
            lambda fut: self.result_callback(fut, generation, purpose))

    def result_callback(self, future, generation=None, purpose=None):
        if generation is not None and (generation != self._nav_request_generation
                                       or generation == self._nav_result_generation):
            return
        self._nav_result_generation = generation
        purpose = purpose or self._nav_goal_purpose
        response = future.result()
        status = response.status
        self.current_handle = None

        if purpose != 'HOME' and (self.is_returning_home
                                  or self.cancel_reason in ('STOP', 'BATTERY_LOW')):
            self.return_home_by_stop()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.abort_retry_count = 0
            if (not self.is_returning_home and self.cancel_reason not in ('STOP', 'BATTERY_LOW')
                    and self._odom.fresh(time.monotonic(), self.get_clock().now().nanoseconds)
                    and self._acquisition_lock.waypoint_succeeded()):
                self._set_recovery_status('COLLECTION_REARMED')
            if purpose == 'HOME':
                self._home_completed = True
                self.is_running = False
                self.publish_robot_state("state", "Stop")
                self.publish_robot_task("PATROL_COMPLETE", "순찰 종료", "", "Task")
                self.get_logger().info('HOME 복귀 완료')
                if rclpy.ok():
                    rclpy.shutdown()
                return

        if status == GoalStatus.STATUS_CANCELED:
            if purpose == 'HOME':
                self.return_home_by_stop()
                return
            if self.cancel_reason in ("STOP", "BATTERY_LOW"):
                self.return_home_by_stop()
                return
            if self.cancel_reason == "OBJECT" and self.object_found:
                self.get_logger().info('⚠️ 이동 취소됨 (물체 감지). recycle_tracking 호출')
                self.cancel_reason = None
                self.launch_recycle_tracking_action()
                return
            self.get_logger().warn(
                '취소 원인을 확인할 수 없어 현재 waypoint를 재시도합니다.'
            )
            self.send_next_goal()
            return

        if status == GoalStatus.STATUS_ABORTED:
            self.abort_retry_count += 1;
            if self.abort_retry_count < self.max_abort_retry:
                if self.is_returning_home:
                    self.return_home_by_stop()
                else:
                    self.send_next_goal()
            else:
                if self.is_returning_home:
                    self.get_logger().error('HOME Goal rejected!')
                    self.publish_robot_state("state", "Error")
                    return

                self.get_logger().warn('Goal rejected! Skipping.')
                self.abort_retry_count = 0
                self.current_idx += 1
                self.send_next_goal()

            return

        if self.current_idx < len(self.waypoints):
            x, y = self.waypoints[self.current_idx]
            self.get_logger().info(f'✅ Reached ({x:.2f}, {y:.2f})')
            
        self.current_idx += 1
        self.send_next_goal()

    def feedback_callback(self, feedback_msg, generation=None):
        if (self.is_returning_home or generation is not None
                and (generation != self._nav_request_generation
                     or generation == self._nav_result_generation)):
            return
        dist = feedback_msg.feedback.distance_remaining
        is_last_waypoint = (
            self.current_idx == len(self.waypoints) - 1
        )

        if is_last_waypoint and not self.object_found:
            if dist <= self.home_arrive_threshold:
                self.object_found = True


def main(args=None):
    rclpy.init(args=args)
    node = AutoNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 사용자에 의해 노드가 정지되었습니다.')
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
