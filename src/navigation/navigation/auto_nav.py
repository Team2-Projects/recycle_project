import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus
from std_msgs.msg import Int32, String
from vision_msgs.msg import Detection2DArray
from collections import Counter
import json
import math
import time

from my_yolo_msgs.msg import DetectedObject
from navigation_interface.action import RecycleActionMsg
from navigation_interface.srv import ControlServo
from navigation_interface.srv import ControlPantilt
from .inference_control import InferenceControl

object_name = {0: 'can', 1: 'paper', 2: 'plastic', 3: 'trash', 4: 'person'}
recyclable_id = {name: idx for idx, name in object_name.items() if name != 'person'}

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')

        self.pending_detection_max_age_sec = float(self.declare_parameter(
            'pending_detection_max_age_sec', 2.0).value)
        if (not math.isfinite(self.pending_detection_max_age_sec)
                or self.pending_detection_max_age_sec <= 0):
            raise ValueError('pending_detection_max_age_sec는 0보다 큰 유한한 값이어야 합니다.')

        self.basket_min_samples = self.declare_parameter('basket_min_samples', 3).value
        self.basket_agreement_ratio = self.declare_parameter('basket_agreement_ratio', 0.8).value
        self.basket_settle_sec = self.declare_parameter('basket_settle_sec', 0.4).value
        if (self.basket_min_samples < 3 or not 0.5 < self.basket_agreement_ratio <= 1.0
                or not 0.0 <= self.basket_settle_sec < 3.0):
            raise ValueError('수거함 재분류: 검출 수 >= 3, 일치율 > 0.5~1, 안정화 시간 0~3초 미만')
        self._basket_votes = Counter()
        self._basket_skips = Counter()
        self._basket_center = None
        self._basket_deadline = self._basket_after_ns = None
        self._basket_last_stamp_ns = self._basket_last_received_ns = 0

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
        self.trigger_servo_movement(0, 0)

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
        self._nav_goal_pending = False
        self._pending_detection = None
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
        self.target_x = None
        self.target_y = None
        self.target_h = None
        self.center_x = None
        self.center_y = None
        self.inference_control = InferenceControl(self)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # 표시용 상태: 현재 검출과 구분하여 실제 수거·하역에 채택한 종류를 알린다.
        self.selected_class_pub = self.create_publisher(
            Int32, '/selected_recycle_class', latched_qos)
        self.publish_selected_class(None)

        self.create_subscription(Path, '/coverage_path', self.path_callback, latched_qos)
        
        self.object_sub = self.create_subscription(
            DetectedObject,
            '/classified_detected_object_info',
            self.object_callback,
            10
        )
        self.create_subscription(
            Detection2DArray, '/yolo/class_observation', self.basket_class_callback, 1)

        self.command_sub = self.create_subscription(
            String,
            "/navigation_command",
            self.command_callback,
            10
        )
    
        self.get_logger().info('AutoNav Ready with Multi-collection, Motor, and Web UI integration.')

    def trigger_servo_movement(self, angle1, angle2):
        req = ControlServo.Request()
        req.angle1 = float(angle1)
        req.angle2 = float(angle2)
        self.servo_future = self.servo_client.call_async(req)

    def trigger_pantilt_movement(self, angle):
        req = ControlPantilt.Request()
        req.angle = float(angle)
        self.pantilt_future = self.pantilt_client.call_async(req)

    def publish_selected_class(self, class_id):
        """늦게 시작한 화면도 마지막 채택 종류를 받으며, -1은 채택 대상 없음."""
        msg = Int32()
        msg.data = -1 if class_id is None else class_id
        self.selected_class_pub.publish(msg)

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
        self._pending_detection = None
        self._basket_deadline = None

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
        self._pending_detection = None
        self._basket_deadline = None
        if self.collected_count == 0:
            self.publish_selected_class(None)
        # 대기 중이던 순찰 재개 콜백도 제거한다. HOME 이동에는 검출이 필요 없다.
        self.inference_control.set_enabled(True)
        if self.cancel_reason == "STOP":
            self.publish_robot_task('USER_COMMAND', '사용자 명령', '순찰 종료', 'Task')
        elif self.cancel_reason == "BATTERY_LOW":
            self.publish_robot_task('BATTERY_LOW', '배터리 경고', '배터리가 30% 이하', 'WARNING')

        self.publish_object_found("-", "-")
        self.publish_robot_state('state', 'Return Home')

        self.cancel_reason = None
        self.object_found = True
        self.is_returning_home = True

        self.object_found = True
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
        self.inference_control.set_enabled(True, self.send_next_goal)

        self.publish_robot_state("state", "Running")
        self.publish_robot_task("PATROL_START", "순찰 시작", "", "Task")

    def object_callback(self, msg):
        if not self.inference_control.ready:
            self._pending_detection = None
            return
        if (msg.id < 0 or not all(math.isfinite(value) for value in msg.coord)
                or msg.coord[2] <= 0 or msg.coord[3] <= 0):
            self._pending_detection = None
            return

        self.target_x = float(msg.coord[0])
        self.target_y = float(msg.coord[1])
        self.target_h = float(msg.coord[3])
        self.object_id = msg.id
        self.y_min = float(getattr(msg, 'min_y', 0)) 
        if (self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending
                or self.is_returning_home):
            self._pending_detection = None
            return
        if self.collected_count > 0 and msg.id != self.previous_object_id:
            self._pending_detection = None
            return 
        
        if self.object_found:
            self.target_x = float(msg.coord[0])
            self.target_y = float(msg.coord[1])
            self.target_h = float(msg.coord[3])
            self.object_id = msg.id
            self.y_min = float(getattr(msg, 'min_y', 0)) 
            return

        if self.current_handle is None:
            if self._nav_goal_pending:
                if self._pending_detection is None:
                    self.get_logger().info('물체 검출 보관: 순찰 이동 요청의 수락을 기다립니다.')
                self._pending_detection = (msg, time.monotonic())
                self.cmd_vel_pub.publish(Twist())
            return

        # 이동 핸들이 확보된 뒤에만 발견 상태로 전환하고 취소를 요청한다.
        self._pending_detection = None
        obj_name = object_name.get(msg.id, '-')
        conf_val = f"{msg.confidence:.2f}" if hasattr(msg, 'confidence') else "1.00"
        self.publish_recycle_success(obj_name, conf_val)

        if self.collected_count == 0:
            self.previous_object_id = msg.id
            self.get_logger().info(f'🎯 Target Object ID set to: {self.previous_object_id}')

        self.publish_selected_class(self.previous_object_id)

        self.object_found = True

        self.get_logger().info("Object detected! Triggering servo...")
        self.trigger_servo_movement(-90, 90)

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
        goal_msg.target_x = float(self.target_x)
        goal_msg.target_y = float(self.target_y)
        goal_msg.target_h = float(self.target_h)
        
        self.get_logger().info('🚀 recycle_tracking_action 호출 (회전 + 접근)')
        future = self._recycle_tracking_client.send_goal_async(goal_msg)
        future.add_done_callback(self.recycle_tracking_goal_response_callback)

    def recycle_tracking_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.resume_after_tracking_failure('수거 요청이 거절됨')
            return

        self.tracking_handle = goal_handle

        if self.stop_pending:
            self.stop_pending = False
            self.tracking_handle.cancel_goal_async()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_tracking_result_callback)

    def recycle_tracking_result_callback(self, future):
        response = future.result()
        status = response.status
        result = response.result
        self.tracking_handle = None

        if (status == GoalStatus.STATUS_CANCELED
                or self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending):
            self.stop_pending = False
            self.trigger_servo_movement(0, 0)
            self.return_home_by_stop()
            return

        if not result.success:
            self.resume_after_tracking_failure(result.message)
            return

        self.get_logger().info("Successed tracking! Triggering servo & pantilt...")
        self.trigger_servo_movement(0, 0)
        self.trigger_pantilt_movement(90)

        self.collected_count += 1
        self.get_logger().info(f'📦 물품 수거 성공! (현재 수거량: {self.collected_count})')
        
        self.y_min = None
        self.get_logger().info('⏳ 3초간 수거함 상태 확인 중...')
        self.check_timer = self.create_timer(3.0, self.check_recycle_condition_callback)
        self._basket_votes.clear()
        self._basket_skips.clear()
        self._basket_center = self._basket_after_ns = None
        self._basket_last_stamp_ns = self._basket_last_received_ns = 0
        self._basket_deadline = time.monotonic() + 3.0 if self.collected_count == 1 else None
        if self.collected_count == 1:
            self.pantilt_future.add_done_callback(self._basket_tilt_done)

    def _basket_tilt_done(self, future):
        """틸트 성공 응답 뒤 안정화 시간을 두고, 이후 PC에서 받은 영상으로 재확인한다."""
        if (future is not self.pantilt_future or self._basket_deadline is None
                or time.monotonic() >= self._basket_deadline):
            return
        try:
            if not future.result().success:
                self.get_logger().warn('수거함 재분류 생략: 틸트 이동 실패')
                return
        except Exception as exc:
            self.get_logger().warn(f'수거함 재분류 생략: 틸트 응답 오류 ({exc})')
            return
        self._basket_after_ns = self.get_clock().now().nanoseconds + int(self.basket_settle_sec * 1e9)
        self.get_logger().info(f'수거함 재분류: 틸트 응답 성공, 안정화 {self.basket_settle_sec:.1f}초')

    def basket_class_callback(self, batch):
        """첫 물체에 한해 같은 위치의 새 프레임을 모으며, 접근 중 분류는 바꾸지 않는다."""
        if (self._basket_deadline is None or time.monotonic() >= self._basket_deadline
                or self.collected_count != 1 or not self.inference_control.ready):
            return
        if self._basket_after_ns is None:
            self._basket_skips['틸트 응답 전'] += 1
            return
        received = batch.header.stamp.sec * 1_000_000_000 + batch.header.stamp.nanosec
        if not self._basket_after_ns < received <= self.get_clock().now().nanoseconds:
            self._basket_skips['안정화 전/PC 시각'] += 1
            return
        if len(batch.detections) != 1 or len(batch.detections[0].results) != 1:
            self._basket_skips['무효 관측'] += 1
            return
        msg = batch.detections[0]
        # 촬영 시각은 중복·역순 확인에만 사용한다. Pi와 PC의 절대 시각은 비교하지 않는다.
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if (received <= self._basket_last_received_ns
                or (stamp > 0 and stamp <= self._basket_last_stamp_ns)):
            self._basket_skips['중복/역순'] += 1
            return
        self._basket_last_received_ns = received
        self._basket_last_stamp_ns = max(self._basket_last_stamp_ns, stamp)
        hypothesis = msg.results[0].hypothesis
        class_id = recyclable_id.get(hypothesis.class_id)
        center = (msg.bbox.center.position.x, msg.bbox.center.position.y)
        if (class_id is None or not 0.0 < hypothesis.score <= 1.0
                or not all(math.isfinite(v) for v in (*center, msg.bbox.size_x, msg.bbox.size_y))
                or min(msg.bbox.size_x, msg.bbox.size_y) <= 0):
            self._basket_skips['무효 관측'] += 1
            return
        # 현재 640x480 영상에서 첫 관측 중심의 50px 이내만 같은 물체로 취급한다.
        if self._basket_center is not None and math.dist(center, self._basket_center) > 50.0:
            self._basket_skips['다른 위치'] += 1
            return
        if self._basket_center is None:
            self._basket_center = center
        self._basket_votes[class_id] += 1

    def _finish_basket_classification(self):
        if self._basket_deadline is None:
            return
        self._basket_deadline = None
        if (self.collected_count != 1 or self.cancel_reason in ('STOP', 'BATTERY_LOW')
                or self.stop_pending):
            return
        total = sum(self._basket_votes.values())
        class_id, count = self._basket_votes.most_common(1)[0] if total else (None, 0)
        old_name = object_name.get(self.previous_object_id, '-')
        if total >= self.basket_min_samples and count / total >= self.basket_agreement_ratio:
            self.previous_object_id = class_id
            self.publish_selected_class(class_id)
            self.get_logger().info(
                f'수거함 분류 확정: {old_name} → {object_name[class_id]} ({count}/{total}회 일치)')
        else:
            reason = ('틸트 성공 응답 없음' if self._basket_after_ns is None
                      else '재분류 관측 수신 없음' if not total and not self._basket_skips
                      else f'제외={dict(self._basket_skips)}')
            self.get_logger().info(
                f'수거함 분류 유지: {old_name} (최다 {count}/{total}회, {reason})')

    def resume_after_tracking_failure(self, reason):
        """서보를 닫힘 위치로 보내고 실패 사유를 남긴 뒤 기존 순찰 목표로 복귀한다."""
        self.get_logger().warn(f'수거 중단: {reason}')
        self.trigger_servo_movement(0, 0)
        self._pending_detection = None
        if self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending:
            self.stop_pending = False
            self.return_home_by_stop()
            return
        self.object_found = False
        self.publish_selected_class(self.previous_object_id if self.collected_count else None)
        self.send_goal(self.resume_x, self.resume_y)

    def _delayed_resume(self):
        self.delay_timer.cancel()
        self.destroy_timer(self.delay_timer)
        self.object_found = False
        self.send_goal(self.resume_x, self.resume_y)

    def check_recycle_condition_callback(self):
        self.check_timer.cancel()
        self.destroy_timer(self.check_timer)
        self._finish_basket_classification()
        if self.cancel_reason in ('STOP', 'BATTERY_LOW') or self.stop_pending:
            self.stop_pending = False
            self.trigger_pantilt_movement(151)
            self.return_home_by_stop()
            return

        if self.y_min is not None and 0 <= self.y_min <= 230:
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
        if self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending:
            self.stop_pending = False
            self.return_home_by_stop()
            return
        self.object_found = True
        self._clear_detection_state()
        self.inference_control.set_enabled(False)
        goal_msg = RecycleActionMsg.Goal()
        # 첫 물체의 수거함 재확인을 반영한 채택 종류로 하역한다.
        goal_msg.index = self.previous_object_id if self.previous_object_id is not None else 1
        goal_msg.current_idx = self.current_idx
        goal_msg.home_x = self.home_x
        goal_msg.home_y = self.home_y

        self.get_logger().info(
            f'분리수거장 이동 요청: 하역 종류 {object_name.get(goal_msg.index, "-")} '
            f'(ID: {goal_msg.index})')
        try:
            future = self._recycle_client.send_goal_async(goal_msg)
        except Exception as exc:
            self._recycle_request_failed(str(exc))
            return
        future.add_done_callback(self.recycle_goal_response_callback)

    def _clear_detection_state(self):
        self._pending_detection = None
        self._basket_deadline = None
        self.target_x = self.target_y = self.target_h = None
        self.object_id = None
        self.y_min = None

    def _recycle_request_failed(self, reason):
        self.get_logger().error(f'분리수거장 이동 요청 실패: {reason}')
        self._clear_detection_state()
        self.inference_control.set_enabled(True)
        if self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending:
            self.stop_pending = False
            self.return_home_by_stop()
            return
        self.object_found = False
        self.publish_robot_state('state', 'Error')

    def recycle_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._recycle_request_failed(str(exc))
            return
        if not goal_handle.accepted:
            self._recycle_request_failed('recycle 목표 거절됨')
            return

        self.recycle_handle = goal_handle

        if self.stop_pending or self.cancel_reason in ("STOP", "BATTERY_LOW"):
            self.stop_pending = False
            self.recycle_handle.cancel_goal_async()
            
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_result_callback)

    def recycle_result_callback(self, future):
        self.recycle_handle = None
        self._clear_detection_state()
        try:
            response = future.result()
        except Exception as exc:
            self.inference_control.set_enabled(True)
            self.get_logger().error(f'분리수거장 이동 결과 수신 실패: {exc}')
            self.publish_robot_state('state', 'Error')
            return
        status = response.status
        result = response.result

        if (status == GoalStatus.STATUS_CANCELED
                or self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending):
            self.stop_pending = False
            self.return_home_by_stop()
            return

        if status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            self.inference_control.set_enabled(True)
            self.get_logger().warn(f'분리수거장 이동 실패')
            self.publish_robot_task('OBJECT_PICKUP_FAIL', '분리수거 실패', '', 'Error')
            if rclpy.ok():
                rclpy.shutdown()
            return

        self.collected_count = 0
        self.previous_object_id = None
        self.publish_selected_class(None)
        # OFF 응답 지연이나 YOLO 재시작 중에는 ON 확인 전 순찰을 보내지 않는다.
        self.get_logger().info('하역 완료: YOLO 추론 재개 확인 후 순찰합니다.')
        self.inference_control.set_enabled(True, self._resume_patrol_after_recycle)

    def _resume_patrol_after_recycle(self):
        self._clear_detection_state()
        self.object_found = False

        self.publish_robot_task("PATROL_RESUME", "순찰 재개", "", "Task")

        self.get_logger().info(f'↩️ 버리기 완료! 원래 목표로 복귀 시작')

        self.current_idx = 0
        self.send_next_goal()

    def send_next_goal(self):
        if self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending:
            self.stop_pending = False
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
        if self.current_idx > len(self.waypoints) - 2:
            self.object_found = True
        self.send_goal(x, y)

    def send_goal(self, x, y):
        # 새 이동 요청에는 이전 요청에서 보관했던 검출을 넘기지 않는다.
        self._pending_detection = None
        self._nav_goal_pending = True
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
        self._nav_goal_pending = False
        pending_detection = self._pending_detection
        self._pending_detection = None
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self.cancel_reason in ("STOP", "BATTERY_LOW") or self.stop_pending:
                self.stop_pending = False
                self.return_home_by_stop()
                return
            if self.is_returning_home:
                self.get_logger().error('HOME Goal rejected!')
                self.publish_robot_state("state", "Error")
                return

            self.get_logger().warn('Goal rejected! Skipping.')
            self.current_idx += 1
            self.send_next_goal()
            return

        self.current_handle = goal_handle

        if self.stop_pending or self.cancel_reason in ("STOP", "BATTERY_LOW"):
            self.stop_pending = False

            self.get_logger().info(
                '대기 중이던 STOP 요청 처리 → NavigateToPose 취소'
            )

            self.current_handle.cancel_goal_async()
        elif pending_detection is not None:
            msg, received = pending_detection
            if time.monotonic() - received <= self.pending_detection_max_age_sec:
                self.object_callback(msg)
            else:
                self.get_logger().info('보관한 검출의 유효 시간이 지나 새 검출을 기다립니다.')
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

        # 취소 요청보다 도착·실패가 먼저 확정되어도 수거/STOP 요청을 잃지 않는다.
        if status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                      GoalStatus.STATUS_ABORTED):
            if self.cancel_reason in ("STOP", "BATTERY_LOW"):
                self.stop_pending = False
                self.trigger_servo_movement(0, 0)
                self.return_home_by_stop()
                return
            if self.cancel_reason == "OBJECT" and self.object_found:
                self.get_logger().info('물체 감지 후 이동 종료 확인: recycle_tracking 호출')
                self.cancel_reason = None
                self.launch_recycle_tracking_action()
                return

        if status == GoalStatus.STATUS_CANCELED:
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
