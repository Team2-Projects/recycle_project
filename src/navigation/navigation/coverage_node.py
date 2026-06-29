import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.ndimage import binary_dilation, label
import numpy as np
import time

# ── 파라미터 ──────────────────────────────────────────
ROBOT_RADIUS_M  = 0.15
SAFETY_MARGIN_M = ROBOT_RADIUS_M + 0.25  # 벽에서 30cm 안전 마진
DILATION_ITER   = 2                       # 장애물 팽창 확대
# ─────────────────────────────────────────────────────

class CoveragePlanner(Node):

    def __init__(self):
        super().__init__('coverage_planner')

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos
        )
        self.path_pub = self.create_publisher(Path, '/coverage_path', qos)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.free_cells     = None
        self.map_info       = None
        self.home_x         = None
        self.home_y         = None
        self.path_published = False  # 중복 발행 방지
        self.map_timer      = None   # 타이머 중복 생성 방지

        self.timer = self.create_timer(1.0, self.try_get_home)

    # ── 맵 수신 ──────────────────────────────────────
    def map_callback(self, msg):
        width  = msg.info.width
        height = msg.info.height
        grid   = np.array(msg.data).reshape(height, width)

        obstacle = np.logical_or(grid >= 50, grid == -1)

        if DILATION_ITER > 0:
            inflated = binary_dilation(obstacle, iterations=DILATION_ITER)
        else:
            inflated = obstacle

        free = np.logical_and(grid == 0, ~inflated)

        labeled, num = label(free)
        if num == 0:
            self.get_logger().warn('No free region found in map!')
            return

        largest   = max(range(1, num + 1), key=lambda i: np.sum(labeled == i))
        free_mask = labeled == largest

        self.free_cells = np.argwhere(free_mask)
        self.map_info   = msg.info
        self.get_logger().info(
            f'Map received: {len(self.free_cells)} free cells  '
            f'(dilation={DILATION_ITER}, margin={SAFETY_MARGIN_M}m)'
        )

    # ── home 위치 획득 ────────────────────────────────
    def try_get_home(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            self.home_x = transform.transform.translation.x
            self.home_y = transform.transform.translation.y
            self.get_logger().info(f'Home saved: ({self.home_x:.2f}, {self.home_y:.2f})')

            self.timer.cancel()
            self.publish_path()

        except TransformException:
            self.get_logger().info('Waiting for map->base_link transform...')

    # ── 안전거리 적용된 후보 셀 반환 ─────────────────
    def get_safe_cells(self):
        resolution = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        rows = self.free_cells[:, 0]
        cols = self.free_cells[:, 1]

        world_xs = cols * resolution + ox
        world_ys = rows * resolution + oy

        x_max = world_xs.max()
        x_min = world_xs.min()
        y_max = world_ys.max()
        y_min = world_ys.min()

        safe_mask = (
            (world_xs <= x_max - SAFETY_MARGIN_M) &
            (world_xs >= x_min + SAFETY_MARGIN_M) &
            (world_ys <= y_max - SAFETY_MARGIN_M) &
            (world_ys >= y_min + SAFETY_MARGIN_M)
        )

        if not np.any(safe_mask):
            self.get_logger().warn('Safety margin fallback: using all free cells.')
            safe_mask = np.ones(len(world_xs), dtype=bool)

        return world_xs[safe_mask], world_ys[safe_mask]

    # ── 공통: 거리 최대 + 보조축 home에 가장 가까운 셀 ──
    def pick_farthest(self, safe_xs, safe_ys, dist_arr, fix_axis='y'):
        tolerance = self.map_info.resolution
        max_val   = dist_arr.max()
        far_mask  = dist_arr >= (max_val - tolerance)

        candidates_x = safe_xs[far_mask]
        candidates_y = safe_ys[far_mask]

        if fix_axis == 'y':
            anchor = np.abs(candidates_y - self.home_y)
        else:
            anchor = np.abs(candidates_x - self.home_x)

        best_idx = np.argmin(anchor)
        return float(candidates_x[best_idx]), float(candidates_y[best_idx])

    # ── 가로 최원점 ───────────────────────────────────
    def get_farthest_x_goal(self, safe_xs, safe_ys):
        dx = np.abs(safe_xs - self.home_x)
        gx, gy = self.pick_farthest(safe_xs, safe_ys, dx, fix_axis='y')
        self.get_logger().info(f'X-farthest: ({gx:.2f}, {gy:.2f})')
        return gx, gy

    # ── 세로 최원점 ───────────────────────────────────
    def get_farthest_y_goal(self, safe_xs, safe_ys):
        dy = np.abs(safe_ys - self.home_y)
        gx, gy = self.pick_farthest(safe_xs, safe_ys, dy, fix_axis='x')
        self.get_logger().info(f'Y-farthest: ({gx:.2f}, {gy:.2f})')
        return gx, gy + 0.2

    # ── 대각선 최원점 ─────────────────────────────────
    def get_farthest_xy_goal(self, safe_xs, safe_ys):
        dx   = np.abs(safe_xs - self.home_x)
        dy   = np.abs(safe_ys - self.home_y)
        dist = dx + dy
        idx  = np.argmax(dist)
        gx, gy = float(safe_xs[idx]), float(safe_ys[idx])
        self.get_logger().info(f'XY-farthest: ({gx:.2f}, {gy:.2f})  dist={dist[idx]:.2f}m')
        return gx -0.2, gy + 0.2

    # ── 홈(Home)과 대각선 끝점(XY)의 딱 중간 점 구하기 ──
    def get_center_goal(self, xy_far_x, xy_far_y):
        target_center_x = (self.home_x + xy_far_x) / 2.0
        target_center_y = (self.home_y + xy_far_y) / 2.0

        return target_center_x, target_center_y

    # ── 경로 발행 ─────────────────────────────────────
    def publish_path(self):
        if self.path_published:
            return

        if self.free_cells is None:
            self.get_logger().info('Map not ready, retrying...')
            if self.map_timer is None:
                self.map_timer = self.create_timer(1.0, self.publish_path)
            return

        if self.map_timer is not None:
            self.map_timer.cancel()
            self.map_timer = None

        safe_xs, safe_ys = self.get_safe_cells()

        goal_x      = self.get_farthest_x_goal(safe_xs, safe_ys)
        goal_y      = self.get_farthest_y_goal(safe_xs, safe_ys)
        goal_xy     = self.get_farthest_xy_goal(safe_xs, safe_ys)
        goal_center = self.get_center_goal(goal_xy[0], goal_xy[1])  # 👈 중앙 좌표 획득
        
        waypoints = [
            goal_x,
            goal_xy,
            goal_center,
            goal_y,
            goal_center,
            (self.home_x, self.home_y)
        ]

        self.get_logger().info('Waypoints (Updated with Center):')
        labels = ['X-farthest', 'XY-farthest', 'CENTER-goal', 'Y-farthest', 'CENTER-goal', 'HOME']
        for lbl, (wx, wy) in zip(labels, waypoints):
            self.get_logger().info(f'  [{lbl}] ({wx:.2f}, {wy:.2f})')

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp    = self.get_clock().now().to_msg()

        for wx, wy in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp    = path.header.stamp
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.path_published = True
        self.get_logger().info(
            f'Path published: {len(waypoints)} waypoints'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CoveragePlanner()

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.path_published:
            break

    node.get_logger().info('waiting short delay for network buffer transmission...')
    time.sleep(1.5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()