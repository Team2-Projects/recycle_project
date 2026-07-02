import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from rclpy.duration import Duration

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException

from navigation_interface.action import RecycleActionMsg

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
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._tick_period = 0.02  # 50Hz
        self._tick_waiters = []   # list of (target_time, future)
        self._tick_timer = self.create_timer(self._tick_period, self._on_tick)

        self.warmup()
        self.get_logger().info("Recycle Done")

    # 노드 생성 직후 TF 버퍼가 채워질 시간을 준다. __init__ 안이라 아직
    # executor.spin()이 돌기 전이므로 여기서는 blocking spin_once로 충분하다.
    def warmup(self):
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < 1.0e9:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _on_tick(self):
        if not self._tick_waiters:
            return
        now = self.get_clock().now()
        remaining = []
        for target_time, future in self._tick_waiters:
            if future.done():
                continue
            if now >= target_time:
                future.set_result(None)
            else:
                remaining.append((target_time, future))
        self._tick_waiters = remaining

    async def _sleep(self, duration: float):
        future = Future()
        target_time = self.get_clock().now() + Duration(seconds=duration)
        self._tick_waiters.append((target_time, future))
        try:
            await future
        finally:
            # 태스크가 취소되는 경우에도 waiter 목록에 남아있지 않도록 정리
            if not future.done():
                future.cancel()

    # 주어진 Twist를 rate_hz 주기로 publish하면서 duration 초 동안 기다린다.
    # 별도 publish용 타이머를 만들지 않고 _sleep으로 쪼개면서 그 사이 publish한다.
    async def _publish_for_duration(self, twist: Twist, duration: float, rate_hz: float = 20.0):
        period = 1.0 / rate_hz
        elapsed = 0.0
        try:
            while elapsed < duration:
                self.cmd_vel_pub.publish(twist)
                await self._sleep(period)
                elapsed += period
        finally:
            self.stop_robot()

    async def execute_callback(self, goal_handle):
        request = goal_handle.request
        self.index = request.index
        self.home_x = request.home_x
        self.home_y = request.home_y
        self.center_x = request.center_x
        self.center_y = request.center_y

        self.recycle_point0_x = self.home_x - 0.3
        self.recycle_point0_y = self.home_y + 0.1
        self.recycle_point1_x = self.home_x - 0.3
        self.recycle_point1_y = self.home_y
        self.recycle_point2_x = self.home_x - 0.3
        self.recycle_point2_y = self.home_y - 0.1

        self.get_logger().info(
            f"Recycle Start: HOME으로 이동 ({self.home_x:.2f}, {self.home_y:.2f})"
        )

        result = RecycleActionMsg.Result()

        if await self.go_home():
            await self.move_backward()
            await self.rotate_180()
            result.success = True
            result.message = "done"
            goal_handle.succeed()
        else:
            self.get_logger().warn('HOME 이동 실패, 후진/회전 스킵')
            result.success = False
            result.message = "home navigation failed"
            goal_handle.abort()

        return result

    async def go_home(self) -> bool:
        try:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            if self.index == 0:
                pose.pose.position.x = self.recycle_point0_x
                pose.pose.position.y = self.recycle_point0_y
            elif self.index == 1:
                pose.pose.position.x = self.recycle_point1_x
                pose.pose.position.y = self.recycle_point1_y
            elif self.index == 2:
                pose.pose.position.x = self.recycle_point2_x
                pose.pose.position.y = self.recycle_point2_y
            else:
                pose.pose.position.x = self.home_x
                pose.pose.position.y = self.home_y

            pose.pose.orientation.w = 1.0

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            self._action_client.wait_for_server()
            self.get_logger().info('1️⃣ wait_for_server 통과')

            goal_handle = await self._action_client.send_goal_async(goal_msg)
            self.get_logger().info(f'2️⃣ goal_handle = {goal_handle}')

            if goal_handle is None:
                self.get_logger().error('❌ goal_handle이 None입니다')
                return False

            if not goal_handle.accepted:
                self.get_logger().warn('HOME goal rejected!')
                return False

            result = await goal_handle.get_result_async()
            self.get_logger().info(f'3️⃣ result = {result}')

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

    async def move_backward(self, duration: float = 2.0, speed: float = -0.2):
        msg = Twist()
        msg.linear.x = speed
        msg.angular.z = 0.0
        await self._publish_for_duration(msg, duration)

    async def rotate_180(self):
        start_yaw = self.get_current_yaw()
        if start_yaw is None:
            self.get_logger().warn('TF 획득 실패, 회전 스킵')
            return

        target_yaw = normalize_angle(start_yaw + math.pi)  # 180도 목표

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.5

        angle_tolerance = math.radians(8.0)  # 8도 오차까지 허용
        control_period = 0.05                # 20Hz 제어 주기
        max_duration = 10.0                  # 안전장치: 10초 넘으면 강제 종료
        elapsed = 0.0

        try:
            while elapsed < max_duration:
                current_yaw = self.get_current_yaw()

                if current_yaw is not None:
                    diff = abs(normalize_angle(target_yaw - current_yaw))
                    if diff <= angle_tolerance:
                        break
                    self.cmd_vel_pub.publish(msg)

                await self._sleep(control_period)
                elapsed += control_period

            if elapsed >= max_duration:
                self.get_logger().warn('회전 타임아웃, 강제 종료')

            self.get_logger().info('180도 회전 완료 (목표 오차 이내)')
        finally:
            self.stop_robot()

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)

    def destroy_node(self):
        # 노드 종료 시 공용 tick 타이머 정리
        try:
            self._tick_timer.cancel()
            self.destroy_timer(self._tick_timer)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Recycle()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()