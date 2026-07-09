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
import time
import math

from my_yolo_cpp_pkg import detected_object_id
from .nav_utils import normalize_angle, get_yaw_from_quaternion
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

ACTION_RECYCLE_NODE = 'action_recycle_node'

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        
        # self.set_nav2_angular_limit(0.1)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._recycle_client = ActionClient(self, RecycleActionMsg, 'recycle_action')
        self._recycle_tracking_client = ActionClient(self, RecycleActionMsg, 'recycle_tracking_action')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.waypoints          = []
        self.current_idx        = 0

        self.is_running = False
        self.object_found = False
        self.objcet_id = None
        self.coord = None
        self.home_x = None
        self.home_y = None
        self.center_x = None
        self.center_y = None

        self.flag = False

        self.resume_x = None
        self.resume_y = None
        self.current_handle = None
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
            detected_object_id.DetectedObject,
            '/detected_object_info',
            self.object_callback,
            10
        )

        self.get_logger().info('AutoNav Ready.')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        self.center_x = self.waypoints[2][0]
        self.center_y = self.waypoints[2][1]

        self.current_idx = 0
        self.is_running = True
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        self.send_next_goal()

    def object_callback(self, msg):
        if self.object_found:
            return

        # -1이면 아무것도 안함
        if msg.id == -1:
            return
        else:
            self.object_found = True

            self.target_x = float(msg.coord[0])
            self.target_y = float(msg.coord[1])
            self.target_h = float(msg.coord[3])
            self.objcet_id = msg.id

            # 재개용 원래 목표 저장
            if self.current_idx < len(self.waypoints):
                self.resume_x, self.resume_y = self.waypoints[self.current_idx]
            
            # [보안] 혹시 모를 충돌을 방지하기 위해 로봇에게 즉시 정지 명령을 먼저 날림
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            # 현재 가던 자율주행 목표 취소
            if self.current_handle is not None:
                self.current_handle.cancel_goal_async()

    # recycle_tracking
    def launch_recycle_tracking_action(self):
        goal_msg = RecycleActionMsg.Goal()
        goal_msg.target_x = self.target_x
        goal_msg.target_y = self.target_y
        goal_msg.target_h = self.target_h
        self.get_logger().info('🚀 recycle_tracking_action 호출 (회전 + 접근)')
        self._recycle_tracking_client.wait_for_server()
        future = self._recycle_tracking_client.send_goal_async(goal_msg)
        future.add_done_callback(self.recycle_tracking_goal_response_callback)

    def recycle_tracking_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.send_goal(self.resume_x, self.resume_y)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_tracking_result_callback)

    def recycle_tracking_result_callback(self, future):
        result = future.result().result

        if not result.success:
            return

        self.launch_recycle_action()

    # recycle
    def launch_recycle_action(self):
        goal_msg = RecycleActionMsg.Goal()
        goal_msg.index = self.objcet_id if self.objcet_id is not None else 1
        goal_msg.current_idx = self.current_idx
        goal_msg.home_x = self.home_x
        goal_msg.home_y = self.home_y
        goal_msg.center_x = self.center_x
        goal_msg.center_y = self.center_y

        self.get_logger().info('🚀 recycle_action 호출 (HOME 이동 + 후진 + 회전)')
        self._recycle_client.wait_for_server()
        future = self._recycle_client.send_goal_async(goal_msg)
        future.add_done_callback(self.recycle_goal_response_callback)

    def recycle_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ recycle 목표 거절됨')
            self.object_found = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.recycle_result_callback)

    def recycle_result_callback(self, future):
        result = future.result().result

        if not result.success:
            self.get_logger().warn(f'Recycle 실패: {result.message}')
            self.object_found = False
            return

        self.object_found = False
        
        self.is_resuming = True
        self.get_logger().info(f'↩️ 원래 목표로 복귀 시작')
        if self.current_idx in (3, 4):
            self.send_goal(self.center_x, self.center_y)
            self.flag = True
        else:
            self.send_goal(self.resume_x, self.resume_y)

    def send_next_goal(self):
        if self.current_idx >= len(self.waypoints):
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

        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = pose

        self._action_client.wait_for_server()
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
        status = future.result().status
        if status == GoalStatus.STATUS_CANCELED:
            if self.object_found:
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
            
        if self.flag:
            self.flag = False
            self.send_goal(self.resume_x, self.resume_y)
        else:
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