import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.ndimage import binary_dilation, label
import numpy as np
import time

class CoveragePlanner(Node):

    def __init__(self):
        super().__init__('coverage_planner')

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.path_pub = self.create_publisher(Path, '/coverage_path', qos)

        self.home_x         = None
        self.home_y         = None
        self.path_published = False  # 중복 발행 방지

    # ── 경로 발행 ─────────────────────────────────────
    def publish_path(self):
        if self.path_published:
            return

        self.home_x = 0.2
        self.home_y = -1.5

        waypoints = [
            (0.6, -2.3), # 오른쪽 아래
            (0.8, -0.5), # 왼쪽 중간
            (2.6, -0.5), # 왼쪽 위
            (2.9, -2.5), # 오른쪽 위
            (1.8, -2.5), # 오른쪽 중간
            (0.9, -2.5), # 오른쪽 아래
            (self.home_x, self.home_y)
        ]

        labels = ['1', '2', '3', '4', '5', '6', 'HOME']
        for lbl, (wx, wy) in zip(labels, waypoints):
            self.get_logger().info(f'  [{lbl}] ({wx:.2f}, {wy:.2f})')

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp    = self.get_clock().now().to_msg()

        for wx, wy in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.path_published = True
            
        self.get_logger().info(
            f'Path published: {len(waypoints)} waypoints'
        )


# coverage_node.py 의 main 함수
def main(args=None):
    rclpy.init(args=args)
    node = CoveragePlanner()
    node.publish_path()

    while rclpy.ok() and not node.path_published:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info('📡 경로 발행 완료!')
    
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()