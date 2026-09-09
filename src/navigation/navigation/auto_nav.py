import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from std_srvs.srv import Trigger
from navigation.tracking_control import Observation, RecoveryCounter
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
        self._sensor_observation = None
        self._sensor_sequence = 0
        recovery_params = {
            'sensor_recovery_frames': 3,
            'sensor_recovery_min_interval_sec': 0.05,
            'sensor_recovery_max_interval_sec': 1.20,
        }
        for name, value in recovery_params.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        self._sensor_recovery = RecoveryCounter(
            int(self.get_parameter('sensor_recovery_frames').value),
            float(self.get_parameter('sensor_recovery_min_interval_sec').value),
            float(self.get_parameter('sensor_recovery_max_interval_sec').value),
        )

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
        self._recycle_tracking_client.wait_for_server()
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.servo_client = self.create_client(ControlServo, 'control_servo')
        # while not self.servo_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for servo service on Raspberry Pi...')

        self.pantilt_client = self.create_client(ControlPantilt, 'control_pantilt')
        # while not self.pantilt_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for pantilt service on Raspberry Pi...')

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
        # 집게가 열린 상태인지 추적한다. 실패 경로에서 반드시 닫기 위함.
        self._servo_open = False
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

        self.resume_sensor_srv = self.create_service(
            Trigger, '/auto_nav/resume_sensor_hold', self.resume_sensor_hold_callback,
        )
        self.reset_tracking_hold_srv = self.create_service(
            Trigger, '/auto_nav/reset_tracking_hold', self.reset_tracking_hold_callback,
        )
        self.get_logger().info('AutoNav Ready with Multi-collection, Motor, and Web UI integration.')

    def trigger_servo_movement(self, angle1, angle2, purpose='', verify=False, retries=1):
        """서보 각도 요청. verify=True 면 ControlServo 응답까지 확인한다.

        기존 코드는 call_async 결과를 보지 않아 서보가 실제로 움직였는지 알 수
        없었다. 집게가 열린 채로 순찰을 계속하면 다음 대상 접근이나 주행에
        지장이 있으므로, 열기/닫기는 응답을 확인하고 실패 시 1회 재시도한다.
        """
        label = purpose or f'angle=({angle1}, {angle2})'
        req = ControlServo.Request()
        req.angle1 = float(angle1)
        req.angle2 = float(angle2)
        if not self.servo_client.service_is_ready():
            self.get_logger().warn(f'서보 서비스 미준비 상태에서 요청 ({label})')
        self.servo_future = self.servo_client.call_async(req)
        if verify:
            self.servo_future.add_done_callback(
                lambda fut: self._servo_result_callback(fut, angle1, angle2, label, retries)
            )
        else:
            # 확인하지 않는 호출도 목표 상태는 기록해 둔다.
            self._servo_open = (float(angle1), float(angle2)) != (0.0, 0.0)
        return self.servo_future

    def _servo_result_callback(self, future, angle1, angle2, label, retries):
        """ControlServo 응답 확인. 실패하면 재시도하고, 그래도 실패하면 알린다."""
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
        """열린 집게를 중립(0, 0)으로 되돌린다. 이미 닫혀 있으면 아무것도 하지 않는다."""
        if not self._servo_open:
            return
        self.get_logger().info(f'집게 닫기 요청 ({reason})')
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

        self.publish_schedule_status("CANCEL")

        if self.tracking_handle is not None:
            self.get_logger().info('Tracking Action 취소 요청')
            self.tracking_handle.cancel_goal_async()
            return

        if self.recycle_handle is not None:
            self.get_logger().info('Recycle Action 취소 요청')
            self.recycle_handle.cancel_goal_async()
            return

        if self.current_handle is not None:
            self.get_logger().info('NavigateToPose 취소 요청')
            self.current_handle.cancel_goal_async()
            return

        self.stop_pending = True

        self.get_logger().warn('현재 취소할 Action이 없습니다.')

    def return_home_by_stop(self):
        if self.cancel_reason == "STOP":
            self.publish_robot_task('USER_COMMAND', '사용자 명령', '순찰 종료', 'Task')
        elif self.cancel_reason == "BATTERY_LOW":
            self.publish_robot_task('BATTERY_LOW', '배터리 경고', '배터리가 30% 이하', 'WARNING')

        self.publish_object_found("-", "-")
        self.publish_robot_state('state', 'Return Home')

        self.cancel_reason = None
        self.object_found = True
        self.is_returning_home = True

        self.close_servo_if_open('STOP/배터리 복귀')

        self.get_logger().info('사용자 STOP 또는 배터리 부족 → HOME 복귀')
        self.send_goal(self.home_x, self.home_y)

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
        self.send_next_goal()

        self.publish_robot_state("state", "Running")
        self.publish_robot_task("PATROL_START", "순찰 시작", "", "Task")

    def _record_sensor_recovery(self, msg):
        """Runs before object_found/id checks, even when safety hold is latched."""
        if self._tracking_hold_reason not in ('SENSOR_STALE', 'VISION_NOT_READY'):
            return
        now = time.monotonic()
        self._sensor_sequence += 1
        try:
            x, y, width, height = (float(v) for v in msg.coord)
            self._sensor_observation = Observation(
                self._sensor_sequence, now, int(msg.id), float(msg.confidence),
                x, y, width, height,
            )
        except (TypeError, ValueError, AttributeError, OverflowError):
            self._sensor_recovery.reset()
            self._sensor_observation = None
            return
        was_ready = self._sensor_recovery.healthy(now)
        ready = self._sensor_recovery.push(now)
        if ready and not was_ready:
            self.get_logger().info(
                '비전 결과 수신 정상화. 주변 확인 후 /auto_nav/resume_sensor_hold 호출로 '
                '동일 대상 재정렬 가능 (순찰을 자동으로 시작하지 않음).')

    def resume_sensor_hold_callback(self, request, response):
        """Operator-confirmed retry of the SAME target after vision recovery."""
        response.success = False
        retryable_reasons = ('SENSOR_STALE', 'VISION_NOT_READY')
        if not self._tracking_safety_hold or self._tracking_hold_reason not in retryable_reasons:
            response.message = (
                'SENSOR_STALE/VISION_NOT_READY 정지에서만 동일 대상 재시도 가능; '
                '기타 홀드는 /auto_nav/reset_tracking_hold 사용'
            )
            return response
        now = time.monotonic()
        if not self._sensor_recovery.healthy(now):
            response.message = '정상 간격의 새 결과 연속 수신이 아직 확인되지 않음'
            return response
        if (self.cancel_reason in ('STOP', 'BATTERY_LOW') or self.is_returning_home
                or not self.is_running or self.tracking_handle is not None
                or self.current_handle is not None or self.recycle_handle is not None):
            response.message = '다른 작업/취소/HOME 복귀 상태이므로 재개하지 않음'
            return response
        obs = self._sensor_observation
        if (self.object_id is None or obs is None
                or not obs.usable(self.object_id, True)
                or not 0 <= now - obs.received_at <= self._sensor_recovery.max_interval):
            response.message = '수신은 복구됐으나 기존 대상의 유효한 최신 좌표가 없음'
            return response
        if not self._recycle_tracking_client.server_is_ready():
            response.message = '추적 Action 서버가 준비되지 않음'
            return response
        self.cmd_vel_pub.publish(Twist())
        # 실패 시 닫았으므로 재시도 전에 다시 연다.
        self.trigger_servo_movement(-90, 90, purpose='열기/추적재시도', verify=True)
        self.target_x, self.target_y, self.target_h = obs.x, obs.y, obs.height
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
        """Explicitly abandon a held target and resume patrol without restarting."""
        response.success = False
        if not self._tracking_safety_hold:
            response.message = '해제할 tracking hold가 없음'
            return response
        if (self.cancel_reason in ('STOP', 'BATTERY_LOW') or self.is_returning_home
                or self.tracking_handle is not None or self.current_handle is not None
                or self.recycle_handle is not None):
            response.message = '다른 작업/취소/HOME 복귀 상태이므로 hold를 해제하지 않음'
            return response

        previous_reason = self._tracking_hold_reason or 'UNKNOWN'
        self.cmd_vel_pub.publish(Twist())
        self.close_servo_if_open('hold 수동 해제')
        self._tracking_safety_hold = False
        self._tracking_hold_reason = ''
        self._sensor_recovery.reset()
        self._sensor_observation = None
        self.object_found = False
        self.object_id = None
        if self.cancel_reason == 'OBJECT':
            self.cancel_reason = None
        self._tracking_retry_after = time.monotonic() + self.tracking_failure_cooldown_sec

        has_resume_goal = (self.is_running and self.resume_x is not None
                           and self.resume_y is not None)
        if has_resume_goal:
            self.publish_robot_task(
                'PATROL_RESUME', '순찰 재개',
                f'수동 hold 해제({previous_reason}); 현재 대상 포기', 'Task'
            )
            self.send_goal(self.resume_x, self.resume_y)
            response.message = f'{previous_reason} hold 해제; 현재 대상 포기 후 순찰 재개 요청'
        else:
            self.publish_robot_task(
                'TRACKING_HOLD_RESET', '정지 hold 해제',
                f'{previous_reason}; 재개할 순찰 목표 없음', 'Warning'
            )
            response.message = f'{previous_reason} hold 해제; 재개할 순찰 목표가 없어 정지 유지'
        response.success = True
        return response

    def _handle_tracking_failure(self, message):
        """Resume simple visual failures; safety holds require explicit operator action."""
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().warn(f'Tracking 종료: {message}')
        # 객체 감지 시 열어둔 집게를 반드시 중립으로 되돌린다. 열린 채로 두면
        # 다음 대상 접근과 주행에 지장이 있다.
        self.close_servo_if_open('추적 실패')
        if self.cancel_reason in ('STOP', 'BATTERY_LOW'):
            self.return_home_by_stop()
            return
        code = message.split(':', 1)[0]
        self._tracking_hold_reason = code
        self._sensor_recovery.reset()
        self._sensor_observation = None
        self._tracking_retry_after = (
            time.monotonic() + self.tracking_failure_cooldown_sec
        )
        self.publish_robot_task('OBJECT_PICKUP_FAIL', '추적 종료', message, 'Warning')
        recoverable = code in ('LOST_TARGET', 'ALIGN_TIMEOUT', 'APPROACH_TIMEOUT')
        has_resume_goal = self.resume_x is not None and self.resume_y is not None
        if not recoverable or not has_resume_goal:
            self._tracking_safety_hold = True
            self.object_found = True
            self.publish_robot_task('TRACKING_HOLD', '정지 확인 필요', message, 'Warning')
            return
        self._tracking_safety_hold = False
        self._tracking_hold_reason = ''
        self.object_found = False
        self.publish_robot_task('PATROL_RESUME', '순찰 재개', '추적 실패 후 복귀', 'Task')
        self.send_goal(self.resume_x, self.resume_y)

    def object_callback(self, msg):
        # 홀드가 걸려 있어도 비전 수신 회복 여부는 계속 관찰한다.
        self._record_sensor_recovery(msg)
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

        self.target_x = values[0]
        self.target_y = values[1]
        self.target_h = values[3]
        self.object_id = msg.id
        self.y_min = float(getattr(msg, 'min_y', 0))
        if self.collected_count > 0 and msg.id != self.previous_object_id:
            return 
        
        if self.object_found:
            return

        # 정지 확인이 필요한 홀드 상태이거나 실패 직후 쿨다운 중이면 재시작하지 않는다.
        if (self._tracking_safety_hold
                or time.monotonic() < self._tracking_retry_after
                or self.cancel_reason in ('STOP', 'BATTERY_LOW')
                or self.is_returning_home or not self.is_running
                or self.current_handle is None):
            return


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
        self.publish_robot_task("OBJECT_PICKUP_START", "수거 시작", "", "Task")
        
        goal_msg = RecycleActionMsg.Goal()
        # index = 추적할 클래스 ID. SetTracking.target_class_id 로 전달되어
        # 인식 노드가 다른 클래스로 대상을 바꾸지 않게 잠근다.
        goal_msg.index = int(self.object_id if self.object_id is not None else 0)
        goal_msg.target_x = float(self.target_x)
        goal_msg.target_y = float(self.target_y)
        goal_msg.target_h = float(self.target_h)
        
        self.get_logger().info('🚀 recycle_tracking_action 호출 (회전 + 접근)')
        future = self._recycle_tracking_client.send_goal_async(goal_msg)
        future.add_done_callback(self.recycle_tracking_goal_response_callback)

    def recycle_tracking_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Tracking 목표 거절됨! 원래 복귀 지점으로 주행')
            self.object_found = False
            self.send_goal(self.resume_x, self.resume_y)
            return

        self.tracking_handle = goal_handle

        if self.stop_pending:
            self.stop_pending = False
            self.tracking_handle.cancel_goal_async()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_tracking_result_callback)

    def recycle_tracking_result_callback(self, future):
        self.tracking_handle = None
        try:
            response = future.result()
            status = response.status
            result = response.result
        except Exception as exc:
            self._handle_tracking_failure(f'INTERNAL_ERROR: 추적 결과 수신 오류: {exc}')
            return

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

        self.get_logger().info("Successed tracking! Triggering servo & pantilt...")
        self.trigger_servo_movement(0, 0, purpose='닫기/수거성공', verify=True)
        self.trigger_pantilt_movement(90)

        self.collected_count += 1
        self.get_logger().info(f'📦 물품 수거 성공! (현재 수거량: {self.collected_count})')
        
        self.y_min = None
        self.get_logger().info('⏳ 3초간 수거함 상태 확인 중...')
        self.check_timer = self.create_timer(3.0, self.check_recycle_condition_callback)

    def _delayed_resume(self):
        self.delay_timer.cancel()
        self.destroy_timer(self.delay_timer)
        self.object_found = False
        self.send_goal(self.resume_x, self.resume_y)

    def check_recycle_condition_callback(self):
        self.check_timer.cancel()
        self.destroy_timer(self.check_timer)

        if self.y_min is not None and 0 <= self.y_min <= 180:
            self.object_found = True  
            self.get_logger().info(f'🗑️ 수거함 포화 감지 (y_min: {self.y_min:.1f})! HOME으로 이동합니다.')
            self.trigger_pantilt_movement(151)
            self.launch_recycle_action()
        else:
            self.trigger_pantilt_movement(151)
            self.get_logger().info('🔄 수거 완료. 순찰을 계속합니다.')
            if self.object_found:
                self.delay_timer = self.create_timer(2.0, self._delayed_resume)
            #     self.object_found = False
            # self.send_goal(self.resume_x, self.resume_y)

    def launch_recycle_action(self):
        goal_msg = RecycleActionMsg.Goal()
        goal_msg.index = self.object_id if self.object_id is not None else 1
        goal_msg.current_idx = self.current_idx
        goal_msg.home_x = self.home_x
        goal_msg.home_y = self.home_y

        self.get_logger().info('🚀 recycle_action 호출 (HOME 이동 + 후진 + 회전)')
        future = self._recycle_client.send_goal_async(goal_msg)
        future.add_done_callback(self.recycle_goal_response_callback)

    def recycle_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ recycle 목표 거절됨')
            self.object_found = False
            return

        self.recycle_handle = goal_handle

        if self.stop_pending:
            self.stop_pending = False
            self.recycle_handle.cancel_goal_async()
            
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_result_callback)

    def recycle_result_callback(self, future):
        response = future.result()
        status = response.status
        result = response.result
        self.recycle_handle = None

        if status == GoalStatus.STATUS_CANCELED:
            self.return_home_by_stop()
            return

        # RecycleActionMsg.Result 에는 type 필드가 없다(success/message 뿐).
        # 기존 result.type 은 AttributeError 를 내며 콜백이 죽어 순찰이 재개되지 않았다.
        if not result.success:
            self.get_logger().warn(f'분리수거장 이동 실패: {result.message}')
            self.publish_robot_task('OBJECT_PICKUP_FAIL', '분리수거 실패', '', 'Error')
            if rclpy.ok():
                rclpy.shutdown()
            return

        self.collected_count = 0
        self.previous_object_id = None
        self.object_found = False

        self.publish_robot_task("PATROL_RESUME", "순찰 재개", "", "Task")

        self.get_logger().info(f'↩️ 버리기 완료! 원래 목표로 복귀 시작')

        self.current_idx = 0
        self.send_next_goal()

    def send_next_goal(self):
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
        self.send_goal(x, y)

    def send_goal(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self.is_returning_home:
                self.get_logger().error('HOME Goal rejected!')
                self.publish_robot_state("state", "Error")
                return

            self.get_logger().warn('Goal rejected! Skipping.')
            self.current_idx += 1
            self.send_next_goal()
            return

        self.current_handle = goal_handle

        if self.stop_pending:
            self.stop_pending = False

            self.get_logger().info(
                '대기 중이던 STOP 요청 처리 → NavigateToPose 취소'
            )

            self.current_handle.cancel_goal_async()
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        response = future.result()
        status = response.status
        self.current_handle = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.abort_retry_count = 0
            if self.is_returning_home:
                self.publish_robot_state("state", "Stop")
                self.publish_robot_task("PATROL_COMPLETE", "순찰 종료", "", "Task")
                self.get_logger().info('HOME 복귀 완료')
                if rclpy.ok():
                    rclpy.shutdown()
                return

        if status == GoalStatus.STATUS_CANCELED:
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
                    self.send_goal(self.home_x, self.home_y)
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

    def feedback_callback(self, feedback_msg):
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