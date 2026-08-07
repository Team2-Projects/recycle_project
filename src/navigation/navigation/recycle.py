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

from action_msgs.msg import GoalStatus


class Recycle(Node):
    def __init__(self):
        super().__init__("recycle")

        self.waypoints = []
        self.current_idx = 0

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
            if not future.done():
                future.cancel()

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
        try:
            result = RecycleActionMsg.Result()
            request = goal_handle.request
            self.index = request.index
            self.current_idx = request.current_idx
            self.home_x = request.home_x
            self.home_y = request.home_y
            # self.center_x = request.center_x
            # self.center_y = request.center_y

            self.recycle_point0 = [
                (-0.3, -0.5),
                (-0.75, -0.5)
            ]
            self.recycle_point1 = [
                (-0.3, -1.0),
                (-0.75, -1.0)
            ]
            self.recycle_point2 = [
                (-0.3, -1.5),
                (-0.75, -1.5)
            ]
            self.recycle_point3 = [
                (-0.3, -2.0),
                (-0.75, -2.0)
            ]
            self.recycle_point4 = [
                (-0.3, -2.5),
                (-0.75, -2.5)
            ]

            if self.index == 0:
                self.waypoints = self.recycle_point0
            elif self.index == 1:
                self.waypoints = self.recycle_point1
            elif self.index == 2:
                self.waypoints = self.recycle_point2
            elif self.index == 3:
                self.waypoints = self.recycle_point3
            elif self.index == 4:
                self.waypoints = self.recycle_point4
            else:
                result.success = False
                result.message = "invalid index"
                goal_handle.abort()
                return result

            for i, (target_x, target_y) in enumerate(self.waypoints):
                success = await self.go_to_pose(target_x, target_y)

                if not success:
                    result.success = False
                    result.message = f"Waypoint {i+1} 이동 실패"
                    goal_handle.abort()
                    return result

            await self.move_backward()
            
            await self.go_to_pose(self.home_x, self.home_y)
                
            result.success = True
            result.message = "done"
            goal_handle.succeed()

            return result
        except Exception as e:
            import traceback
            self.get_logger().error(traceback.format_exc())

            result = RecycleActionMsg.Result()
            result.success = False
            result.message = str(e)
            goal_handle.abort()
            return result

    async def go_to_pose(self, x: float, y: float) -> bool:
        try:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            self._action_client.wait_for_server()

            goal_handle = await self._action_client.send_goal_async(goal_msg)

            if goal_handle is None:
                self.get_logger().error('❌ goal_handle이 None입니다')
                return False

            if not goal_handle.accepted:
                self.get_logger().warn('HOME goal rejected!')
                return False

            result = await goal_handle.get_result_async()

            status = result.status
            if status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(f'HOME 이동 실패 (status={status})')
                return False
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

    async def move_backward(self, duration: float = 3.0, speed: float = -0.1):
        msg = Twist()
        msg.linear.x = speed
        msg.angular.z = 0.0
        await self._publish_for_duration(msg, duration)

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

    executor = rclpy.executors.SingleThreadedExecutor()
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