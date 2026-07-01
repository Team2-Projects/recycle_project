import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException
from navigation_interface.action import RecycleActionMsg
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math
import time

from .nav_utils import normalize_angle, get_yaw_from_quaternion


class Recycle(Node):
    def __init__(self):
        super().__init__("recycle")

        self.cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            RecycleActionMsg,
            'recycle_action',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.warmup()
        self.get_logger().info("Recycle Done")

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        self.home_x = request.home_x
        self.home_y = request.home_y
        self.center_x = request.center_x
        self.center_y = request.center_y

        self.recycle_point0_x = self.home_x - 1.0
        self.recycle_point0_y = self.home_y + 0.1
        self.recycle_point1_x = self.home_x - 1.0
        self.recycle_point1_y = self.home_y
        self.recycle_point2_x = self.home_x - 1.0
        self.recycle_point2_y = self.home_y - 0.1

        self.get_logger().info(f"Recycle Start: HOME으로 이동 ({self.home_x:.2f}, {self.home_y:.2f})")

        result = RecycleActionMsg.Result()

        if self.go_home():
            self.move_backward()
            self.rotate_timed(5.0)
            result.success = True
            result.message = "done"
            goal_handle.succeed()
        else:
            self.get_logger().warn('HOME 이동 실패, 후진/회전 스킵')
            result.success = False
            result.message = "home navigation failed"
            goal_handle.abort()

        return result

    def warmup(self):
        start_time = time.time()
        while time.time() - start_time < 1.0:
            rclpy.spin_once(self, timeout_sec=0.1)

    def go_home(self) -> bool:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.home_x
        pose.pose.position.y = self.home_y
        pose.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self._nav_client.wait_for_server()
        send_goal_future = self._nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        nav_goal_handle = send_goal_future.result()

        if nav_goal_handle is None or not nav_goal_handle.accepted:
            self.get_logger().warn('HOME goal rejected!')
            return False

        result_future = nav_goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info('✅ HOME 도착')
        return True

    def get_current_yaw(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return get_yaw_from_quaternion(t.transform.rotation)
        except TransformException:
            return None

    def move_backward(self):
        msg = Twist()
        msg.linear.x = -0.2
        msg.angular.z = 0.0

        start_time = time.time()
        while time.time() - start_time < 2.0:
            self.cmd_vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.stop_robot()

    def rotate_timed(self, duration=3.0):
        self.get_logger().info(f'시간 기반 회전 시작: {duration}초 동안 회전')
        
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.5  # 회전 속도 (필요시 조절 가능, 대략 0.5rad/s)

        start_time = time.time()
        while time.time() - start_time < duration:
            if not rclpy.ok():
                break
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05)  # 20Hz 주기로 속도 명령 발행

        self.stop_robot()
        time.sleep(0.3)  # 로봇이 완전히 멈출 때까지 잠시 대기
        self.get_logger().info(f'{duration}초 회전 완료 및 정지')

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Recycle()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()