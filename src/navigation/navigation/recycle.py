import math
import traceback
import json

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from rclpy.duration import Duration
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException

from navigation_interface.action import RecycleActionMsg

from .nav_utils import get_yaw_from_quaternion

from action_msgs.msg import GoalStatus
from std_msgs.msg import String

from navigation_interface.srv import ControlServo


class Recycle(Node):

    def __init__(self):
        super().__init__("recycle")

        self.waypoints = []
        self.current_idx = 0
        self.nav_goal_handle = None

        self.cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            RecycleActionMsg,
            'recycle_action',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
            callback_group=self.cb_group
        )

        self.servo_client = self.create_client(ControlServo, 'control_servo')

        self.get_logger().info("Nav2 Action Server 대기 중...")
        self._action_client.wait_for_server()
        self.get_logger().info("Nav2 Action Server 연결 완료")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self._tick_period = 0.02  # 50Hz
        self._tick_waiters = []

        self._tick_timer = self.create_timer(
            self._tick_period,
            self._on_tick,
            callback_group=self.cb_group
        )

        goal_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.goal_pub = self.create_publisher(PoseStamped, '/navigation_goal', goal_qos)

        self.robot_task_pub = self.create_publisher(String, "/robot_task", 10)

    def publish_robot_task(self, eventType, message, note, status):
        msg = String()
        msg.data = json.dumps({
            "eventType": eventType,
            "message": message,
            "note": note,
            "status": status
        })
        self.robot_task_pub.publish(msg)

    def cancel_callback(self, goal_handle):

        self.get_logger().warn(
            "🛑 Recycle Action cancel 요청 수신"
        )

        self.stop_robot()

        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        return CancelResponse.ACCEPT

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
                remaining.append(
                    (target_time, future)
                )

        self._tick_waiters = remaining

    async def _sleep(self, duration: float):
        future = Future()
        target_time = (
            self.get_clock().now() + Duration(seconds=duration)
        )

        self._tick_waiters.append(
            (target_time, future)
        )

        try:
            await future
        finally:
            if not future.done():
                future.cancel()

    async def _publish_for_duration(self, goal_handle, twist: Twist, duration: float, rate_hz: float = 20.0):

        period = 1.0 / rate_hz
        elapsed = 0.0

        try:
            while elapsed < duration:
                if goal_handle.is_cancel_requested:
                    self.stop_robot()
                    return False
                self.cmd_vel_pub.publish(twist)
                await self._sleep(period)
                elapsed += period
            return True
        finally:
            self.stop_robot()

    async def execute_callback(self, goal_handle):
        result = RecycleActionMsg.Result()
        try:
            request = goal_handle.request

            self.index = request.index
            self.current_idx = request.current_idx
            self.home_x = request.home_x
            self.home_y = request.home_y

            recycle_points = [
                [
                    (-0.3, -0.5),
                    (-0.8, -0.5)
                ],
                [
                    (-0.3, -1.2),
                    (-0.8, -1.2)
                ],
                [
                    (-0.3, -1.8),
                    (-0.8, -1.8)
                ],
                [
                    (-0.3, -2.5),
                    (-0.8, -2.5)
                ]
            ]

            if (self.index < 0 or self.index >= len(recycle_points)):
                result.success = False
                result.message = "invalid index"
                goal_handle.abort()
                return result

            self.waypoints = recycle_points[self.index]

            for i, (target_x, target_y) in enumerate(self.waypoints):
                self.get_logger().info(
                    f"🚗 Waypoint {i + 1} 이동: "
                    f"({target_x}, {target_y})"
                )
                success = await self.go_to_pose(goal_handle, target_x, target_y)

                if goal_handle.is_cancel_requested:
                    return self.cancel_result(goal_handle)

                if not success:
                    result.success = False
                    result.message = (
                        f"Waypoint {i + 1} 이동 실패"
                    )
                    goal_handle.abort()
                    return result

            self.trigger_servo_movement(-90, 90)

            if len(self.waypoints) < 2:
                result.success = False
                result.message = "직전 waypoint가 없습니다"
                goal_handle.abort()
                return result

            previous_x, previous_y = self.waypoints[-2]

            self.get_logger().info(
                f"🔙 직전 waypoint로 후진 시작: "
                f"({previous_x:.3f}, {previous_y:.3f})"
            )

            success = await self.move_backward(goal_handle, previous_x, previous_y, speed=0.08)

            self.publish_robot_task("OBJECT_PICKUP_SUCCESS", "수거 성공", "", "Task")

            if goal_handle.is_cancel_requested:
                return self.cancel_result(goal_handle)
            if not success:
                result.success = False
                result.message = "직전 좌표로 후진 실패"
                goal_handle.abort()
                return result

            self.get_logger().info(
                f"🏠 HOME 이동: "
                f"({self.home_x}, {self.home_y})"
            )

            self.trigger_servo_movement(0, 0)

            success = await self.go_to_pose(goal_handle, self.home_x, self.home_y)

            if goal_handle.is_cancel_requested:
                return self.cancel_result(goal_handle)

            if not success:
                result.success = False
                result.message = "HOME 이동 실패"
                goal_handle.abort()
                return result

            result.success = True
            result.message = "done"

            goal_handle.succeed()

            self.get_logger().info(
                "✅ Recycle 작업 완료"
            )

            return result

        except Exception as e:
            self.stop_robot()
            self.get_logger().error(
                f"Recycle execute exception: {e}"
            )
            self.get_logger().error(
                traceback.format_exc()
            )
            if goal_handle.is_cancel_requested:
                return self.cancel_result(goal_handle)

            result = RecycleActionMsg.Result()

            result.success = False
            result.message = str(e)

            goal_handle.abort()

            return result

    def cancel_result(self, goal_handle):
        self.stop_robot()
        self.nav_goal_handle = None
        goal_handle.canceled()
        result = RecycleActionMsg.Result()
        result.success = False
        result.message = "STOP"

        self.get_logger().warn(
            "🛑 Recycle Action 취소 완료"
        )

        return result

    async def go_to_pose(self, goal_handle, x: float, y: float) -> bool:
        try:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = (self.get_clock().now().to_msg())
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            self.goal_pub.publish(pose)

            self.nav_goal_handle = (
                await self._action_client.send_goal_async(goal_msg)
            )

            if self.nav_goal_handle is None:
                self.get_logger().error(
                    "❌ goal_handle이 None입니다"
                )
                return False

            if not self.nav_goal_handle.accepted:
                self.get_logger().warn(
                    "❌ Nav2 goal rejected!"
                )

                self.nav_goal_handle = None
                return False

            if goal_handle.is_cancel_requested:
                await self.nav_goal_handle.cancel_goal_async()
                self.nav_goal_handle = None
                return False

            result = (
                await self.nav_goal_handle.get_result_async()
            )

            self.nav_goal_handle = None

            status = result.status

            if status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(
                    f"Nav2 이동 실패 "
                    f"(status={status})"
                )
                return False

            self.get_logger().info(
                f"✅ Nav2 도착: "
                f"x={x:.3f}, y={y:.3f}"
            )

            return True

        except Exception as e:
            self.get_logger().error(
                f"❌ go_to_pose 예외 발생: {e}"
            )
            self.get_logger().error(
                traceback.format_exc()
            )
            self.nav_goal_handle = None

            return False

    def get_current_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y

            yaw = get_yaw_from_quaternion(
                t.transform.rotation
            )

            return x, y, yaw

        except TransformException as e:
            self.get_logger().warn(
                f"현재 위치 조회 실패: {e}"
            )
            return None

    def get_current_yaw(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            return get_yaw_from_quaternion(
                t.transform.rotation
            )

        except TransformException:

            return None

    def trigger_servo_movement(self, angle1, angle2):
        req = ControlServo.Request()
        req.angle1 = float(angle1)
        req.angle2 = float(angle2)
        self.servo_future = self.servo_client.call_async(req)

    async def move_backward(
        self,
        goal_handle,
        target_x: float,
        target_y: float,
        speed: float = 0.08
    ):
        msg = Twist()

        tolerance = 0.05
        angle_tolerance = 0.1

        try:
            start_time = self.get_clock().now()

            while rclpy.ok():

                elapsed = (
                    self.get_clock().now() - start_time
                ).nanoseconds / 1e9

                if elapsed > 10.0:
                    self.get_logger().error("❌ 후진 시간 초과")
                    self.stop_robot()
                    return False

                if goal_handle.is_cancel_requested:
                    self.stop_robot()
                    return False

                current_pose = self.get_current_pose()

                if current_pose is None:
                    self.stop_robot()
                    await self._sleep(0.05)
                    continue

                current_x, current_y, current_yaw = current_pose

                dx = target_x - current_x
                dy = target_y - current_y

                distance = math.sqrt(dx * dx + dy * dy)

                # 목표 도착
                if distance <= tolerance:
                    self.stop_robot()
                    return True

                # 목표 방향
                target_angle = math.atan2(dy, dx)

                # 후진이므로 목표 반대 방향을 바라봄
                desired_yaw = target_angle + math.pi

                angle_error = desired_yaw - current_yaw

                angle_error = math.atan2(
                    math.sin(angle_error),
                    math.cos(angle_error)
                )

                # =========================
                # 1단계: 방향 먼저 맞추기
                # =========================
                if abs(angle_error) > angle_tolerance:

                    msg.linear.x = 0.0

                    msg.angular.z = max(
                        -0.5,
                        min(0.5, angle_error)
                    )

                # =========================
                # 2단계: 방향이 맞으면 후진
                # =========================
                else:

                    msg.linear.x = -abs(speed)

                    # 작은 방향 오차만 보정
                    msg.angular.z = max(
                        -0.2,
                        min(0.2, -angle_error)
                    )

                self.cmd_vel_pub.publish(msg)

                await self._sleep(0.02)

        finally:
            self.stop_robot()

    def stop_robot(self):
        stop_msg = Twist()

        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0

        self.cmd_vel_pub.publish(
            stop_msg
        )

    def destroy_node(self):
        try:
            self._tick_timer.cancel()
            self.destroy_timer(
                self._tick_timer
            )

        except Exception:
            pass

        try:
            self.stop_robot()
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