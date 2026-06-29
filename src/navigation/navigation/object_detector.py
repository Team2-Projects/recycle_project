import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
import math

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.pose_pub    = self.create_publisher(PoseStamped, '/object_pose', qos)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.detected      = False
        self.detect_thresh = 2.0  # 👈 2미터 이내 감지

        self.detection_radius = 0.8

        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('Object Detector Ready.')

    def scan_callback(self, msg):
        if self.detected:
            return

        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_x   = transform.transform.translation.x
            robot_y   = transform.transform.translation.y
            tf_time   = transform.header.stamp

            q   = transform.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        except TransformException:
            return

        # 전체 스캔에서 2미터 이내 유효한 값 탐색
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < self.detect_thresh):
                continue

            # 해당 인덱스의 각도 계산
            angle = msg.angle_min + i * msg.angle_increment

            # 정면 ±20도만 (옆 벽 무시)
            if abs(angle) > math.radians(20):
                continue

            # 장애물 좌표 계산 (로봇 기준 → map 기준)
            obj_x = robot_x + r * math.cos(yaw + angle)
            obj_y = robot_y + r * math.sin(yaw + angle)

            distance = math.sqrt((obj_x - robot_x)**2 + (obj_y - robot_y)**2)

            self.get_logger().info(f"{distance}")

            if distance <= self.detection_radius:
                self.get_logger().info(f'🤖 Robot near target center! Distance: {distance:.2f}m')
                # 가제보 시간(tf_time)을 그대로 담아서 발행합니다.
                self.publish_object_pose(obj_x, obj_y, tf_time)

            return  # 첫 번째 감지된 물체만 처리

    def publish_object_pose(self, x, y, stamp):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp    = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.z = 2.0

        for _ in range(3):
            self.pose_pub.publish(pose)

        self.detected = True
        self.get_logger().info(f'🎯 Object pose published: ({x:.2f}, {y:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)

if __name__ == '__main__':
    main()