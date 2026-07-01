import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import Empty
from navigation_interface.srv import RecycleActionMsg
from action_msgs.msg import GoalStatus
import subprocess 
import time
import math
from .nav_utils import normalize_angle, get_yaw_from_quaternion

ACTION_RECYCLE_NODE = 'action_recycle_node'

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_detector_pub = self.create_publisher(Empty, '/reset_object_detector', 10)

        self.waypoints          = []
        self.current_idx        = 0

        self.is_running = False
        self.object_found = False
        self.home_x = None
        self.home_y = None
        self.center_x = None
        self.center_y = None
        self.resume_x = None
        self.resume_y = None
        self.current_handle = None

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.create_subscription(Path, '/coverage_path', self.path_callback, latched_qos)
        self.create_subscription(PoseStamped, '/object_pose', self.object_callback, latched_qos)
        
        self.client = self.create_client(RecycleSrv, "recycle_service")
        
        self.get_logger().info('AutoNav Ready.')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def rotate_180(self):
        start_yaw = self.get_current_yaw()
        if start_yaw is None:
            self.get_logger().warn('TF 획득 실패, 회전 스킵')
            return

        target_yaw = normalize_angle(start_yaw + math.pi)  # 180도 목표

        msg = Twist()
        msg.linear.x = 0.0       # 제자리 회전이므로 전진/후진 없음
        msg.angular.z = 0.5      # 반시계 방향 (라디안/초)

        angle_tolerance = math.radians(3.0)  # 3도 오차까지 허용
        timeout = time.time() + 10.0         # 안전장치: 10초 넘으면 강제 종료

        while rclpy.ok():
            current_yaw = self.get_current_yaw()
            if current_yaw is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            diff = abs(normalize_angle(target_yaw - current_yaw))

            if diff <= angle_tolerance:
                break

            if time.time() > timeout:
                self.get_logger().warn('회전 타임아웃, 강제 종료')
                break

            self.cmd_vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.stop_robot()
        self.get_logger().info(f'180도 회전 완료 (목표 오차 이내)')

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
            return self.home_x, self.home_y  # fallback

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

        self.object_found = True
        self.get_logger().info('🎯 Object detected! 원래 목표 저장 후 이동 취소')

        # 재개용 원래 목표 저장
        self.resume_x, self.resume_y = self.waypoints[self.current_idx]
        
        if self.current_handle is not None:
            self.current_handle.cancel_goal_async()

    def launch_recycle_action_node(self):
        req = RecycleSrv.Request()
        req.index = self.current_idx
        req.home_x = self.home_x
        req.home_y = self.home_y
        req.center_x = self.center_x
        req.center_y = self.center_y

        self.get_logger().info('🚀 Calling recycle_service (HOME 이동 + 후진 + 회전)')
        self.client.wait_for_service()
        future = self.client.call_async(req)
        future.add_done_callback(self.check_recycle_action_done)

    def check_recycle_action_done(self, future):
        res = future.result()

        if not res.success:
            self.get_logger().warn(f'Recycle 실패: {res.message}')
            return
        
        self.get_logger().info("Recycle 끝")
        self.object_found = False
        self.reset_detector_pub.publish(Empty())
        self.get_logger().info('📡 Published /reset_object_detector')

        self.get_logger().info(f'↩️ 원래 목표로 복귀: ({self.resume_x:.2f}, {self.resume_y:.2f})')
        self.send_goal(self.resume_x, self.resume_y)

    def send_next_goal(self):
        if self.current_idx >= len(self.waypoints):
            self.get_logger().info('🏁 Patrol finished. Shutting down...')
            self.destroy_node()
            rclpy.shutdown()
            return

        x, y = self.waypoints[self.current_idx]
        total = len(self.waypoints)
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
                self.launch_recycle_action_node()
            return

        x, y = self.waypoints[self.current_idx]
        self.get_logger().info(f'✅ Reached ({x:.2f}, {y:.2f})')
        if self.current_idx in (1, 3):
            self.get_logger().info('🔄 Waypoint 1 도착, 180도 회전 시작')
            self.rotate_180()
        self.current_idx += 1

        self.send_next_goal()

    def feedback_callback(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        self.get_logger().info(f'  Distance remaining: {dist:.2f}m', throttle_duration_sec=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = AutoNav()
    rclpy.spin(node)

if __name__ == '__main__':
    main()