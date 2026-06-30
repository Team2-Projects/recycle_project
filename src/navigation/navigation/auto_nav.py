import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import Empty
import subprocess 
import time
import math

ACTION_RECYCLE_NODE = 'action_recycle_node'

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_detector_pub = self.create_publisher(Empty, '/reset_object_detector', 10)

        self.original_waypoints = []
        self.waypoints          = []
        self.current_idx        = 0

        # object 처리용 별도 경로 [object_pos, home_pos]
        self.object_waypoints   = []
        self.object_idx         = 0

        self.is_running         = False
        self.object_found       = False
        self.recycle_number = None
        self.home_x             = None
        self.home_y             = None
        self.recycle_point0_x = None
        self.recycle_point0_y = None
        self.recycle_point1_x = None
        self.recycle_point1_y = None
        self.recycle_point2_x = None
        self.recycle_point2_y = None
        self.center_x = None
        self.center_y = None
        self.current_handle     = None

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.create_subscription(Path, '/coverage_path', self.path_callback, latched_qos)
        self.create_subscription(PoseStamped, '/object_pose', self.object_callback, latched_qos)
        self.get_logger().info('AutoNav Ready.')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        self.original_waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.waypoints = list(self.original_waypoints)
        self.home_x = self.waypoints[-1][0]
        self.home_y = self.waypoints[-1][1]

        self.recycle_point0_x = self.home_x - 1.0
        self.recycle_point0_y = self.home_y + 0.1
        self.recycle_point1_x = self.home_x - 1.0
        self.recycle_point1_y = self.home_y
        self.recycle_point2_x = self.home_x - 1.0
        self.recycle_point2_y = self.home_y - 0.1

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
        self.recycle_number = int(msg.pose.orientation.z)
        obj_x = msg.pose.position.x
        obj_y = msg.pose.position.y
        self.get_logger().info(f'🎯 Object detected! Heading to object then home...')

        resume_x, resume_y = self.waypoints[self.current_idx]

        # 💡 [버그 정정] 원래 코드에 recycle_point2_x 자리에 _y가 들어가있던 오타를 고쳤습니다.
        if self.current_idx == 3:
            if self.recycle_number == 0:
                self.object_waypoints = [
                    (obj_x, obj_y, None),
                    (self.center_x, self.center_y, None),
                    (self.recycle_point0_x, self.recycle_point0_y, ACTION_RECYCLE_NODE),
                    (self.center_x, self.center_y, None),
                    (resume_x, resume_y, None), 
                ]
            elif self.recycle_number == 1:
                self.object_waypoints = [
                    (obj_x, obj_y, None),
                    (self.center_x, self.center_y, None),
                    (self.recycle_point1_x, self.recycle_point1_y, ACTION_RECYCLE_NODE),
                    (self.center_x, self.center_y, None),
                    (resume_x, resume_y, None), 
                ]
            elif self.recycle_number == 2:
                self.object_waypoints = [
                    (obj_x, obj_y, None),
                    (self.center_x, self.center_y, None),
                    (self.recycle_point2_x, self.recycle_point2_y, ACTION_RECYCLE_NODE), # ← 정정
                    (self.center_x, self.center_y, None),
                    (resume_x, resume_y, None), 
                ]
        else:
            if self.recycle_number == 0:
                self.object_waypoints = [
                    (obj_x, obj_y, None),
                    (self.recycle_point0_x, self.recycle_point0_y, ACTION_RECYCLE_NODE),
                    (resume_x, resume_y, None), 
                ]
            elif self.recycle_number == 1:
                self.object_waypoints = [
                    (obj_x, obj_y, None),
                    (self.recycle_point1_x, self.recycle_point1_y, ACTION_RECYCLE_NODE),
                    (resume_x, resume_y, None), 
                ]
            elif self.recycle_number == 2:
                self.object_waypoints = [
                    (obj_x, obj_y, None),
                    (self.recycle_point2_x, self.recycle_point2_y, ACTION_RECYCLE_NODE), # ← 정정
                    (resume_x, resume_y, None), 
                ]

        self.object_idx = 0
        
        if self.current_handle is not None:
            self.current_handle.cancel_goal_async()

    def launch_recycle_action_node(self):
        cmd = ['ros2', 'run', 'recycle', 'recycle']
        self.get_logger().info(f'🚀 Launching: {" ".join(cmd)}')
        self._ext_proc  = subprocess.Popen(cmd)
        self._ext_timer = self.create_timer(0.5, self.check_recycle_action_done)

    def check_recycle_action_done(self):
        if self._ext_proc is None:
            return
        if self._ext_proc.poll() is None:
            return

        ret = self._ext_proc.poll()
        self.get_logger().info(f'✅ External node finished (exit={ret}). Resuming.')
        self._ext_timer.cancel()
        self._ext_timer = None
        self._ext_proc  = None

        self.reset_detector_pub.publish(Empty())
        self.get_logger().info('📡 Published /reset_object_detector')

        self.send_next_goal()

    def send_next_goal(self):
        if self.object_found:
            if self.object_idx >= len(self.object_waypoints):
                self.get_logger().info(f'✅ Object handling done. Resuming patrol from waypoint [{self.current_idx}/{len(self.waypoints)}]')
                self.object_found = False
                self.object_waypoints = []
                self.object_idx = 0
                self.current_idx += 1
            else:
                x, y, action = self.object_waypoints[self.object_idx]
                label = '[OBJECT]' if self.object_idx == 0 else '[HOME]'
                self.get_logger().info(f'Navigating to {label} ({x:.2f}, {y:.2f})')
                self.send_goal(x, y)
                return

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
        
        # 🎯 [YAW 수정 핵심] 국룰 실물 트릭 적용
        # 목적지에 들어설 때 특정 각도를 강제하지 않도록 설정합니다.
        # 기존의 math.sin/cos 기반 쿼터니언 변환을 지우고 w=1.0으로 오픈합니다.
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
            if self.object_found:
                self.object_idx += 1
            else:
                self.current_idx += 1
            self.send_next_goal()
            return

        self.current_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        if self.object_found:
            x, y, action = self.object_waypoints[self.object_idx]
            self.get_logger().info(f'✅ Reached object waypoint ({x:.2f}, {y:.2f})')
            self.object_idx += 1

            if action == ACTION_RECYCLE_NODE:
                self.launch_recycle_action_node()
                return
        else:
            x, y = self.waypoints[self.current_idx]
            self.get_logger().info(f'✅ Reached ({x:.2f}, {y:.2f})')
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