import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
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

        # /object_pose 토픽 발행
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', qos)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.detected = False
        self.target_x = None
        self.target_y = None
        
        # 🚨 감지가 너무 안 된다면 반경을 0.5m에서 0.8m로 살짝 넓혀서 확실히 밟게 만듭니다.
        self.detection_radius = 0.6

        self.create_subscription(Path, '/coverage_path', self.path_callback, qos)
        self.timer = self.create_timer(0.2, self.check_robot_position)
        
        self.get_logger().info('Object Detector [Fix Version] Ready.')

    def path_callback(self, msg):
        if self.target_x is not None:
            return

        if len(msg.poses) < 3:
            self.get_logger().warn('Path has less than 3 waypoints!')
            return

        wp2_x = msg.poses[1].pose.position.x
        wp2_y = msg.poses[1].pose.position.y
        wp3_x = msg.poses[2].pose.position.x
        wp3_y = msg.poses[2].pose.position.y

        base_x = (wp2_x + wp3_x) / 2.0
        base_y = (wp2_y + wp3_y) / 2.0

        self.target_x = base_x
        self.target_y = base_y -0.3

        self.get_logger().info(f'📍 Target Center calculated: ({self.target_x:.2f}, {self.target_y:.2f})')

    def check_robot_position(self):
        if self.target_x is None or self.detected:
            return

        try:
            # 시뮬레이션 시간 동기화를 위해 현재 가장 최신의 TF 스탬프를 가져옴
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            tf_time = transform.header.stamp  # 👈 핵심: 가제보 시간 받아오기

            distance = math.sqrt((self.target_x - robot_x)**2 + (self.target_y - robot_y)**2)

            if distance <= self.detection_radius:
                self.get_logger().info(f'🤖 Robot near target center! Distance: {distance:.2f}m')
                # 가제보 시간(tf_time)을 그대로 담아서 발행합니다.
                self.publish_object_pose(self.target_x, self.target_y, tf_time)

        except TransformException:
            pass

    def publish_object_pose(self, x, y, stamp):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp    = stamp  # 👈 가제보 시뮬레이션 시간 매핑
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        # 확실하게 전송되도록 약간의 간격을 두고 3번 연속 펍(Publish)합니다.
        for _ in range(3):
            self.pose_pub.publish(pose)
            
        self.detected = True
        self.get_logger().info(f'🎯 Virtual Object published to /object_pose: ({x:.2f}, {y:.2f})')
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)

if __name__ == '__main__':
    main()