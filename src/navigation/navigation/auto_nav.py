import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import Empty
from navigation_interface.action import RecycleActionMsg
from action_msgs.msg import GoalStatus
from std_msgs.msg import String

import time
import math
import json

from my_yolo_msgs.msg import DetectedObject
from .nav_utils import normalize_angle, get_yaw_from_quaternion
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

ACTION_RECYCLE_NODE = 'action_recycle_node'

object_name = {0: 'can', 1: 'paper', 2: 'plastic'}
class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        
        # self.set_nav2_angular_limit(0.1)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._recycle_client = ActionClient(self, RecycleActionMsg, 'recycle_action')
        self._recycle_tracking_client = ActionClient(self, RecycleActionMsg, 'recycle_tracking_action')
        self._action_client.wait_for_server()
        self._recycle_client.wait_for_server()
        self._recycle_tracking_client.wait_for_server()
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.waypoints          = []
        self.current_idx        = 0

        self.is_running = False
        self.object_found = False
        self.object_id = None
        self.object_msg = None
        self.coord = None
        self.home_x = None
        self.home_y = None
        self.center_x = None
        self.center_y = None

        self.cancel_reason = None
        self.is_returning_home = False

        self.resume_x = None
        self.resume_y = None
        self.current_handle = None
        self.tracking_handle = None
        self.recycle_handle = None
        self.is_resuming = False  

        self.going_home = False  
        self.home_arrive_threshold = 0.5

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

        self.get_logger().info('AutoNav Ready.')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.object_found_pub = self.create_publisher(
            String,
            "/object_found",
            10
        )

        self.robot_status_pub = self.create_publisher(
            String,
            "/robot_status",
            10
        )

        self.robot_task_pub = self.create_publisher(
            String,
            "/robot_task",
            10
        )

        self.command_sub = self.create_subscription(
            String,
            "/navigation_command",
            self.command_callback,
            10
        )

        self.publish_robot_state(
            "state",
            "Running"
        )
        
        self.publish_robot_task(
            "PATROL_START",
            "순찰 시작",
            "",
            "Task"
        )

        self.recycle_success_pub = self.create_publisher(
            String,
            "/recycle_success",
            10
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/navigation_goal',
            10
        )

    def publish_recycle_success(self, object_name, confidence):
        msg = String()
        msg.data = json.dumps({
            "object_name": object_name,
            "confidence": confidence,
            "status": "Success"
        })
        self.recycle_success_pub.publish(msg)

    def publish_object_found(self, object_name, confidence):
        msg = String()
        msg.data = json.dumps({
            "object_name": object_name,
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

    def get_current_yaw(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return get_yaw_from_quaternion(t.transform.rotation)
        except Exception:
            return None

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except:
            return self.home_x, self.home_y  

    def path_callback(self, msg):
        if self.is_running:
            self.get_logger().warn('Already navigating, ignoring new path')
            return

        self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.home_x = self.waypoints[-1][0]
        self.home_y = self.waypoints[-1][1]
        # self.center_x = 1.8
        # self.center_y = -1.5

        self.current_idx = 0
        self.is_running = True
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        self.send_next_goal()

    def return_home_by_stop(self):
        if self.cancel_reason == "STOP":
            self.publish_robot_task(
                'USER_COMMAND',
                '사용자 명령',
                '순찰 종료',
                'Task'
            )
        elif self.cancel_reason == "BATTERY_LOW":
            self.publish_robot_task(
                'BATTERY_LOW',
                '배터리 경고',
                '배터리가 30% 이하',
                'WARNING'
            )

        self.publish_object_found(
            "-",
            "-"
        )

        self.publish_robot_state(
            'state',
            'Return Home'
        )

        self.cancel_reason = None
        self.object_found = True
        self.is_resuming = False
        self.is_returning_home = True

        self.get_logger().info('사용자 STOP → HOME 복귀')

        self.send_goal(self.home_x, self.home_y)
    
    def object_callback(self, msg):
        if self.object_found:
            return

        # -1이면 아무것도 안함
        if msg.id == -1:
            return
        else:
            self.object_found = True

            self.publish_object_found(
                object_name.get(msg.id, '-'),
                f"{msg.confidence:.2f}"
            )

            self.object_msg = msg

            self.publish_robot_task(
                "OBJECT_DETECTED",
                "물체 감지",
                f"물체: {object_name.get(msg.id, '-')} / 신뢰도: {msg.confidence:.2f}",
                "Detect"
            )

            self.target_x = float(msg.coord[0])
            self.target_y = float(msg.coord[1])
            self.target_h = float(msg.coord[3])
            self.object_id = msg.id

            # 재개용 원래 목표 저장
            if self.current_idx < len(self.waypoints):
                self.resume_x, self.resume_y = self.waypoints[self.current_idx]
            
            # [보안] 혹시 모를 충돌을 방지하기 위해 로봇에게 즉시 정지 명령을 먼저 날림
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            # 현재 가던 자율주행 목표 취소
            if self.current_handle is not None:
                self.cancel_reason = "OBJECT"
                self.current_handle.cancel_goal_async()

    def launch_recycle_tracking_action(self):
        self.publish_robot_task(
            "OBJECT_PICKUP_START",
            "수거 시작",
            "",
            "Task"
        )
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
            self.get_logger().warn(f'Tracking 실패: {result.message}')
            self.object_found = False
            self.send_goal(self.resume_x, self.resume_y)
            return

        self.launch_recycle_action()

    # recycle
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

            self.publish_robot_task(
                'OBJECT_PICKUP_FAIL',
                '분리수거 실패',
                '',
                'Error'
            )

            self.object_found = False
            return
        
        # DB저장용 데이터 넘김
        self.publish_recycle_success(
            object_name.get(self.object_msg.id, '-'),
            f"{self.object_msg.confidence:.2f}"
        )

        self.publish_robot_task(
            "OBJECT_PICKUP_SUCCESS",
            "수거 성공",
            "",
            "Task"
        )
        self.publish_robot_task(
            "PATROL_RESUME",
            "순찰 재개",
            "",
            "Task"
        )

        self.object_found = False

        self.object_msg = None
        
        self.is_resuming = True
        self.get_logger().info(f'↩️ 원래 목표로 복귀 시작')
        self.send_goal(self.resume_x, self.resume_y)

    def send_next_goal(self):
        if self.current_idx >= len(self.waypoints):
            self.publish_robot_state(
                "state",
                "Stop"
            )
            self.get_logger().info('🏁 Patrol finished. Shutting down...')
            self.destroy_node()
            rclpy.shutdown()
            return

        x, y = self.waypoints[self.current_idx]
        total = len(self.waypoints)
        self.going_home = (self.current_idx == total - 1)
        label = '[HOME]' if self.current_idx == total - 1 else f'[{self.current_idx + 1}/{total}]'
        self.get_logger().info(f'Navigating to {label} ({x:.2f}, {y:.2f})')
        self.send_goal(x, y)

    def send_goal(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        
        self.goal_pub.publish(pose)

        goal_msg      = NavigateToPose.Goal()
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
                self.publish_robot_state(
                    "state",
                    "Stop"
                )
                self.publish_robot_task(
                    "PATROL_COMPLETE",
                    "순찰 종료",
                    "",
                    "Task"
                )
                self.get_logger().info('HOME 복귀 완료')
                self.destroy_node()
                rclpy.shutdown()
                return

        if status == GoalStatus.STATUS_CANCELED:
            if self.cancel_reason == "STOP" or self.cancel_reason == "BATTERY_LOW":
                self.return_home_by_stop()
                return
            if self.cancel_reason == "OBJECT" and self.object_found:
                self.get_logger().info('⚠️ 이동 취소됨 (물체 감지). recycle 서비스 호출')
                self.launch_recycle_tracking_action()
                return

        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error("Goal Aborted by Nav2")
            self.send_next_goal()
            return

        if self.is_resuming:
            self.get_logger().info('✅ 끊겼던 지점으로 복귀 완료! 다음 웨이포인트로 주행을 이어갑니다.')
            self.is_resuming = False
        else:
            x, y = self.waypoints[self.current_idx]
            self.get_logger().info(f'✅ Reached ({x:.2f}, {y:.2f})')

        self.current_idx += 1
        self.send_next_goal()

    def feedback_callback(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        if self.going_home and not self.object_found:
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()