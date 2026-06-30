import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException
import math
import time


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_yaw_from_quaternion(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Recycle(Node):
    def __init__(self):
        super().__init__("recycle")

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # TF가 채워질 시간을 잠깐 줌 (노드 생성 직후엔 버퍼가 비어있을 수 있음)
        self.warmup()

        self.get_logger().info("Recycle Start")

        self.move_backward()
        self.rotate_180()

        self.get_logger().info("Recycle Done")

    def warmup(self):
        start_time = time.time()
        while time.time() - start_time < 1.0:
            rclpy.spin_once(self, timeout_sec=0.1)

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

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()