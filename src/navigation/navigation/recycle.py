import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException
from navigation_interface.srv import RecycleActionMsg
import math
import time

from .nav_utils import normalize_angle, get_yaw_from_quaternion


class Recycle(Node):
    def __init__(self):
        super().__init__("recycle")

        self.cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            Recycle,
            'recycle_action',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

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
            self.rotate_180()
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
        try:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = self.home_x
            pose.pose.position.y = self.home_y
            pose.pose.orientation.w = 1.0

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            self._action_client.wait_for_server()
            self.get_logger().info('1️⃣ wait_for_server 통과')

            send_goal_future = self._action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future)
            self.get_logger().info('2️⃣ send_goal_future 완료')

            goal_handle = send_goal_future.result()
            self.get_logger().info(f'3️⃣ goal_handle = {goal_handle}')

            if goal_handle is None:
                self.get_logger().error('❌ goal_handle이 None입니다')
                return False

            if not goal_handle.accepted:
                self.get_logger().warn('HOME goal rejected!')
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            self.get_logger().info('4️⃣ result_future 완료')

            result = result_future.result()
            self.get_logger().info(f'5️⃣ result = {result}')

            self.get_logger().info('✅ HOME 도착')
            return True

        except Exception as e:
            self.get_logger().error(f'❌ go_home 예외 발생: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

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

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Recycle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()