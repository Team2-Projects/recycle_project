import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus
from std_msgs.msg import String
import json
import time
import math

from my_yolo_msgs.msg import DetectedObject
from navigation_interface.action import RecycleActionMsg
from navigation_interface.srv import ControlServo
from navigation_interface.srv import ControlPantilt

object_name = {0: 'can', 1: 'paper', 2: 'plastic', 3: 'trash', 4: 'glass_bottle', 5: 'person'}

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._recycle_client = ActionClient(self, RecycleActionMsg, 'recycle_action')
        self._recycle_tracking_client = ActionClient(self, RecycleActionMsg, 'recycle_tracking_action')
        
        self._action_client.wait_for_server()
        self._recycle_client.wait_for_server()
        self._recycle_tracking_client.wait_for_server()
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.servo_client = self.create_client(ControlServo, 'control_servo')
        while not self.servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for servo service on Raspberry Pi...')

        self.pantilt_client = self.create_client(ControlPantilt, 'control_pantilt')
        while not self.pantilt_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for pantilt service on Raspberry Pi...')

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
        self.tracking_handle = None
        self.recycle_handle = None

        self.going_home = False  
        self.home_arrive_threshold = 0.5

        self.collected_count = 0         
        self.previous_object_id = None   
        self.nearest_target_y_up = 0

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
            10
        )

        self.object_found_pub = self.create_publisher(String, "/object_found", 10)
        self.robot_status_pub = self.create_publisher(String, "/robot_status", 10)
        self.robot_task_pub = self.create_publisher(String, "/robot_task", 10)
        self.recycle_success_pub = self.create_publisher(String, "/recycle_success", 10)

        goal_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.goal_pub = self.create_publisher(PoseStamped, '/navigation_goal', goal_qos)

        self.command_sub = self.create_subscription(
            String,
            "/navigation_command",
            self.command_callback,
            10
        )

        self.publish_robot_state("state", "Running")
        self.publish_robot_task("PATROL_START", "순찰 시작", "", "Task")
    
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

    def command_callback(self, msg):
        if msg.data == "STOP":
            self.cancel_reason = "STOP"
        elif msg.data == "BATTERY_LOW":
            self.cancel_reason = "BATTERY_LOW"
        else:
            return

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

        self.get_logger().info('사용자 STOP 또는 배터리 부족 → HOME 복귀')
        self.send_goal(self.home_x, self.home_y)

    def path_callback(self, msg):
        if self.is_running:
            self.get_logger().warn('Already navigating, ignoring new path')
            return

        self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.home_x = self.waypoints[-1][0]
        self.home_y = self.waypoints[-1][1]

        self.current_idx = 0
        self.is_running = True
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        self.send_next_goal()

    def object_callback(self, msg):
        if msg.id == -1:
            return

        self.nearest_target_y_up = float(getattr(msg, 'max_y_up', 0))

        if self.object_found:
            return
            
        if self.collected_count > 0 and msg.id != self.previous_object_id:
            return 

        obj_name = object_name.get(msg.id, '-')
        conf_val = f"{msg.confidence:.2f}" if hasattr(msg, 'confidence') else "1.00"
        self.publish_recycle_success(obj_name, conf_val)

        if self.collected_count == 0:
            self.previous_object_id = msg.id
            self.get_logger().info(f'🎯 Target Object ID set to: {self.previous_object_id}')

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

        self.target_x = float(msg.coord[0])
        self.target_y = float(msg.coord[1])
        self.target_h = float(msg.coord[3])
        self.object_id = msg.id

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
        goal_msg.target_x = self.target_x
        goal_msg.target_y = self.target_y
        goal_msg.target_h = self.target_h
        
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
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_tracking_result_callback)

    def recycle_tracking_result_callback(self, future):
        response = future.result()
        status = response.status
        result = response.result
        self.tracking_handle = None

        if status == GoalStatus.STATUS_CANCELED:
            self.return_home_by_stop()
            return

        if not result.success:
            self.get_logger().warn('Tracking 접근 실패!')
            self.object_found = False
            self.send_goal(self.resume_x, self.resume_y)
            return

        self.get_logger().info("Successed tracking! Triggering servo & pantilt...")
        self.trigger_servo_movement(0, 0)
        self.trigger_pantilt_movement(90)

        self.collected_count += 1
        self.get_logger().info(f'📦 물품 수거 성공! (현재 수거량: {self.collected_count})')
        
        self.get_logger().info('⏳ 3초간 수거함 상태 확인 중...')
        self.check_timer = self.create_timer(3.0, self.check_recycle_condition_callback)

    def check_recycle_condition_callback(self):
        self.check_timer.cancel()
        self.destroy_timer(self.check_timer)

        if 0 <= self.nearest_target_y_up <= 180:
            self.object_found = True  
            self.get_logger().info(f'🗑️ 수거함 포화 감지 (y_up: {self.nearest_target_y_up:.1f})! HOME으로 이동합니다.')
            self.trigger_pantilt_movement(151)
            self.launch_recycle_action()
        else:
            self.trigger_pantilt_movement(151)
            self.get_logger().info('🔄 수거 완료. 순찰을 계속합니다.')
            if self.object_found:
                self.object_found = False
            self.send_goal(self.resume_x, self.resume_y)

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

        if not result.success:
            self.get_logger().warn(f'Recycle 실패: {result.message}')
            self.publish_robot_task('OBJECT_PICKUP_FAIL', '분리수거 실패', '', 'Error')
            self.object_found = False
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
                self.launch_recycle_action()
                return 
            
            self.publish_robot_state("state", "Stop")
            self.publish_robot_task("PATROL_COMPLETE", "순찰 종료", "", "Task")
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

        self.goal_pub.publish(pose)

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
            self.get_logger().warn('Goal rejected! Skipping.')
            self.current_idx += 1
            self.send_next_goal()
            return

        self.current_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        response = future.result()
        status = response.status
        self.current_handle = None

        if status == GoalStatus.STATUS_SUCCEEDED:
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
                self.launch_recycle_tracking_action()
                return

        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error("Goal Aborted by Nav2")
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