import math
import traceback

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from rclpy.duration import Duration
from rclpy.action import ActionClient, ActionServer, CancelResponse

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException

from navigation_interface.action import RecycleActionMsg

from .nav_utils import get_yaw_from_quaternion

from action_msgs.msg import GoalStatus
from std_msgs.msg import String


class Recycle(Node):

    def __init__(self):
        super().__init__("recycle")

        self.waypoints = []
        self.current_idx = 0
        self.nav_goal_handle = None

        self.cb_group = ReentrantCallbackGroup()

        # ============================================================
        # Recycle Action Server
        # ============================================================
        self._action_server = ActionServer(
            self,
            RecycleActionMsg,
            'recycle_action',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        # ============================================================
        # cmd_vel
        # ============================================================
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # ============================================================
        # Nav2 Action Client
        # ============================================================
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
            callback_group=self.cb_group
        )

        self.get_logger().info("Nav2 Action Server 대기 중...")
        self._action_client.wait_for_server()
        self.get_logger().info("Nav2 Action Server 연결 완료")

        # ============================================================
        # TF
        # ============================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # 공용 tick timer
        # ============================================================
        self._tick_period = 0.02  # 50Hz
        self._tick_waiters = []

        self._tick_timer = self.create_timer(
            self._tick_period,
            self._on_tick,
            callback_group=self.cb_group
        )

        # ============================================================
        # Navigation Goal Publisher
        # ============================================================
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/navigation_goal',
            10
        )

    # ================================================================
    # Action Cancel
    # ================================================================
    def cancel_callback(self, goal_handle):

        self.get_logger().warn(
            "🛑 Recycle Action cancel 요청 수신"
        )

        self.stop_robot()

        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        return CancelResponse.ACCEPT

    # ================================================================
    # Tick
    # ================================================================
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

    # ================================================================
    # Sleep
    # ================================================================
    async def _sleep(self, duration: float):

        future = Future()

        target_time = (
            self.get_clock().now()
            + Duration(seconds=duration)
        )

        self._tick_waiters.append(
            (target_time, future)
        )

        try:
            await future

        finally:

            if not future.done():
                future.cancel()

    # ================================================================
    # Publish Twist for Duration
    # ================================================================
    async def _publish_for_duration(
        self,
        goal_handle,
        twist: Twist,
        duration: float,
        rate_hz: float = 20.0
    ):

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

    # ================================================================
    # Execute Callback
    # ================================================================
    async def execute_callback(self, goal_handle):

        result = RecycleActionMsg.Result()

        try:

            request = goal_handle.request

            self.index = request.index
            self.current_idx = request.current_idx
            self.home_x = request.home_x
            self.home_y = request.home_y

            # ========================================================
            # 재활용 수거 지점
            # ========================================================
            recycle_points = [

                [
                    (-0.3, -0.5),
                    (-0.75, -0.5)
                ],

                [
                    (-0.3, -1.0),
                    (-0.75, -1.0)
                ],

                [
                    (-0.3, -1.5),
                    (-0.75, -1.5)
                ],

                [
                    (-0.3, -2.0),
                    (-0.75, -2.0)
                ],

                [
                    (-0.3, -2.5),
                    (-0.75, -2.5)
                ]
            ]

            # ========================================================
            # Index 확인
            # ========================================================
            if (
                self.index < 0
                or self.index >= len(recycle_points)
            ):

                result.success = False
                result.message = "invalid index"

                goal_handle.abort()

                return result

            self.waypoints = recycle_points[self.index]

            self.get_logger().info(
                f"Recycle index = {self.index}"
            )

            self.get_logger().info(
                f"Waypoints = {self.waypoints}"
            )

            # ========================================================
            # 수거 지점 이동
            # ========================================================
            for i, (target_x, target_y) in enumerate(
                self.waypoints
            ):

                self.get_logger().info(
                    f"🚗 Waypoint {i + 1} 이동: "
                    f"({target_x}, {target_y})"
                )

                success = await self.go_to_pose(
                    goal_handle,
                    target_x,
                    target_y
                )

                # ----------------------------------------------------
                # Cancel
                # ----------------------------------------------------
                if goal_handle.is_cancel_requested:
                    return self.cancel_result(goal_handle)

                # ----------------------------------------------------
                # Navigation 실패
                # ----------------------------------------------------
                if not success:

                    result.success = False

                    result.message = (
                        f"Waypoint {i + 1} 이동 실패"
                    )

                    goal_handle.abort()

                    return result

            # ========================================================
            # 직전 좌표로 후진
            # ========================================================

            if len(self.waypoints) < 2:

                result.success = False
                result.message = "직전 waypoint가 없습니다"

                goal_handle.abort()

                return result

            # 마지막 waypoint의 바로 이전 waypoint
            previous_x, previous_y = self.waypoints[-2]

            self.get_logger().info(
                f"🔙 직전 waypoint로 후진 시작: "
                f"({previous_x:.3f}, {previous_y:.3f})"
            )

            success = await self.move_backward(
                goal_handle,
                previous_x,
                previous_y,
                speed=0.08
            )

            if goal_handle.is_cancel_requested:
                return self.cancel_result(goal_handle)

            if not success:
                result.success = False
                result.message = "직전 좌표로 후진 실패"
                goal_handle.abort()
                return result

            
            # ========================================================
            # HOME으로 이동
            # ========================================================

            self.get_logger().info(
                f"🏠 HOME 이동: "
                f"({self.home_x}, {self.home_y})"
            )

            success = await self.go_to_pose(
                goal_handle,
                self.home_x,
                self.home_y
            )

            # --------------------------------------------------------
            # Cancel
            # --------------------------------------------------------
            if goal_handle.is_cancel_requested:
                return self.cancel_result(goal_handle)

            # --------------------------------------------------------
            # HOME 이동 실패
            # --------------------------------------------------------
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

        # ============================================================
        # Exception
        # ============================================================
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

    # ================================================================
    # Cancel Result
    # ================================================================
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

    # ================================================================
    # Navigate To Pose
    # ================================================================
    async def go_to_pose(
        self,
        goal_handle,
        x: float,
        y: float
    ) -> bool:

        try:

            pose = PoseStamped()

            pose.header.frame_id = 'map'

            pose.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            pose.pose.position.x = x
            pose.pose.position.y = y

            # orientation 지정하지 않고
            # 기존 코드처럼 yaw = 0
            pose.pose.orientation.w = 1.0

            goal_msg = NavigateToPose.Goal()

            goal_msg.pose = pose

            # 외부 visualization용
            self.goal_pub.publish(pose)

            self.get_logger().info(
                f"Nav2 Goal 전송: "
                f"x={x:.3f}, y={y:.3f}"
            )

            # --------------------------------------------------------
            # Nav2 Goal 전송
            # --------------------------------------------------------
            self.nav_goal_handle = (
                await self._action_client.send_goal_async(
                    goal_msg
                )
            )

            # --------------------------------------------------------
            # Goal Handle 확인
            # --------------------------------------------------------
            if self.nav_goal_handle is None:

                self.get_logger().error(
                    "❌ goal_handle이 None입니다"
                )

                return False

            # --------------------------------------------------------
            # Goal reject
            # --------------------------------------------------------
            if not self.nav_goal_handle.accepted:

                self.get_logger().warn(
                    "❌ Nav2 goal rejected!"
                )

                self.nav_goal_handle = None

                return False

            # --------------------------------------------------------
            # Cancel 확인
            # --------------------------------------------------------
            if goal_handle.is_cancel_requested:

                await self.nav_goal_handle.cancel_goal_async()

                self.nav_goal_handle = None

                return False

            # --------------------------------------------------------
            # Nav2 결과 대기
            # --------------------------------------------------------
            result = (
                await self.nav_goal_handle
                .get_result_async()
            )

            self.nav_goal_handle = None

            status = result.status

            # --------------------------------------------------------
            # 성공 여부
            # --------------------------------------------------------
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

    # ================================================================
    # Get Current Pose
    # ================================================================
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

    # ================================================================
    # Get Current Yaw
    # ================================================================
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

    # ================================================================
    # Move Backward To Previous Waypoint
    # ================================================================
    async def move_backward(
        self,
        goal_handle,
        target_x: float,
        target_y: float,
        speed: float = 0.08
    ):
        """
        현재 위치에서 target_x까지 직선 후진한다.

        후진 완료 조건:
            current_x >= target_x

        현재 경로는 모두
            x = -0.75 → x = -0.30
        방향이므로 X 좌표만 기준으로 판단한다.

        target_y는 로그 확인용으로만 사용한다.
        """

        self.get_logger().info(
            f"🔙 X 좌표 기준 후진 시작 "
            f"(목표 X = {target_x:.3f})"
        )

        msg = Twist()

        # ------------------------------------------------------------
        # 후진
        # ------------------------------------------------------------
        msg.linear.x = -abs(speed)

        # 회전하지 않음
        msg.angular.z = 0.0

        try:

            while rclpy.ok():

                # ====================================================
                # Cancel 확인
                # ====================================================
                if goal_handle.is_cancel_requested:

                    self.stop_robot()

                    return False

                # ====================================================
                # 현재 위치 확인
                # ====================================================
                current_pose = self.get_current_pose()

                if current_pose is None:

                    self.stop_robot()

                    await self._sleep(0.05)

                    continue

                current_x, current_y, current_yaw = current_pose

                # ====================================================
                # 로그
                # ====================================================
                self.get_logger().debug(
                    f"후진 중 | "
                    f"현재 X={current_x:.3f} | "
                    f"목표 X={target_x:.3f} | "
                    f"Y={current_y:.3f} | "
                    f"Yaw={math.degrees(current_yaw):.1f}°"
                )

                # ====================================================
                # X 좌표 도착 확인
                #
                # 예:
                # 목표 X = -0.30
                #
                # -0.75 < -0.30 → 계속 후진
                # -0.50 < -0.30 → 계속 후진
                # -0.31 < -0.30 → 계속 후진
                # -0.30 >= -0.30 → 정지
                # -0.29 >= -0.30 → 정지
                # ====================================================
                if current_x >= target_x:

                    self.stop_robot()

                    self.get_logger().info(
                        f"✅ 후진 완료 "
                        f"(현재 X={current_x:.3f}, "
                        f"목표 X={target_x:.3f})"
                    )

                    return True

                # ====================================================
                # 직선 후진
                # ====================================================
                msg.linear.x = -abs(speed)

                # 회전하지 않음
                msg.angular.z = 0.0

                self.cmd_vel_pub.publish(msg)

                # 50Hz
                await self._sleep(0.02)

        finally:

            # 어떤 경우에도 로봇 정지
            self.stop_robot()

    # ================================================================
    # Stop Robot
    # ================================================================
    def stop_robot(self):

        stop_msg = Twist()

        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0

        self.cmd_vel_pub.publish(
            stop_msg
        )

    # ================================================================
    # Destroy Node
    # ================================================================
    def destroy_node(self):

        # ------------------------------------------------------------
        # Tick Timer 정리
        # ------------------------------------------------------------
        try:

            self._tick_timer.cancel()

            self.destroy_timer(
                self._tick_timer
            )

        except Exception:
            pass

        # ------------------------------------------------------------
        # Robot 정지
        # ------------------------------------------------------------
        try:
            self.stop_robot()
        except Exception:
            pass

        super().destroy_node()


# ====================================================================
# Main
# ====================================================================
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


# ====================================================================
# Entry Point
# ====================================================================
if __name__ == "__main__":
    main()