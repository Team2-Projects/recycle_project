import rclpy
import asyncio
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from my_yolo_msgs.msg import DetectedObject
from navigation_interface.action import RecycleActionMsg


class RecycleTrackingNode(Node):
    def __init__(self):
        super().__init__('recycle_tracking_node')

        self.cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            RecycleActionMsg,
            'recycle_tracking_action',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group
        )

        # 제어 플래그
        self.align_completed = False
        self.action_finished = False

        # 1. 퍼블리셔 및 구독자 설정
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # YOLO 노드가 보내주는 토픽 구독 (같은 콜백 그룹으로 묶어야 async 태스크와 동시 처리 가능)
        self.yolo_sub = self.create_subscription(
            DetectedObject,
            '/detected_object_info',
            self.yolo_callback,
            10,
            callback_group=self.cb_group
        )

        self.get_logger().info("♻️ 실시간 분리수거 정렬 및 4초 직진 노드가 시작되었습니다.")

    # -------------------------------------------------
    # rclpy에는 진짜 asyncio 이벤트 루프가 없어서, 콜백 안에서
    # await로 "잠깐 쉬기"를 하려면 Future + one-shot timer로 흉내내야 함
    # -------------------------------------------------
    async def _sleep(self, seconds: float):
        future = rclpy.task.Future()

        def _on_timer():
            if not future.done():
                future.set_result(None)

        timer = self.create_timer(seconds, _on_timer, callback_group=self.cb_group)
        try:
            await future
        finally:
            timer.cancel()
            self.destroy_timer(timer)

    async def execute_callback(self, goal_handle):
        self.action_finished = False
        self.align_completed = False

        self.get_logger().info("🎬 recycle_tracking_action 시작: 물체 정렬 대기 중...")

        while not self.action_finished:
            await asyncio.sleep(0.05)

        goal_handle.succeed()

        result = RecycleActionMsg.Result()
        result.success = True
        return result

    def yolo_callback(self, msg):
        # 정렬이 이미 완전히 끝났거나(또는 직진 태스크 진행 중) 로봇을 제어하지 않음
        if self.align_completed:
            return

        msg_cmd = Twist()
        msg_cmd.linear.x = 0.0  # 정렬 중에는 직진하지 않음

        # 1. 물체가 인식되었을 경우 (id가 -1이 아님)
        if msg.id != -1:
            x = msg.coord[0]  # 물체의 중심 X 좌표 추출

            # [규칙 1] X값이 300보다 작으면 물체가 왼쪽에 있으므로 왼쪽으로 회전
            if x < 300.0:
                msg_cmd.angular.z = 0.2
                self.get_logger().info(
                    f"🔍 물체 왼쪽 감지 (X: {x:.1f}) -> 왼쪽으로 회전 중...",
                    throttle_duration_sec=0.5
                )
                self.cmd_vel_pub.publish(msg_cmd)

            # [규칙 2] X값이 350보다 크면 물체가 오른쪽에 있으므로 오른쪽으로 회전
            elif x > 350.0:
                msg_cmd.angular.z = -0.2
                self.get_logger().info(
                    f"🔍 물체 오른쪽 감지 (X: {x:.1f}) -> 오른쪽으로 회전 중...",
                    throttle_duration_sec=0.5
                )
                self.cmd_vel_pub.publish(msg_cmd)

            # [규칙 3] X값이 300 ~ 350 사이면 정중앙에 위치한 것이므로 멈춤!
            else:
                msg_cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(msg_cmd)
                self.get_logger().info(f"🎯 정렬 완료! 물체가 중앙에 있습니다. (X: {x:.1f})")

                # 정렬 플래그를 참으로 만들고, 블로킹 없이 코루틴을 태스크로 던짐
                self.align_completed = True
                asyncio.ensure_future(self.launch_recycle_action())

        # 2. 물체가 인식되지 않았다면 안전을 위해 로봇을 멈춤
        else:
            msg_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(msg_cmd)

    async def launch_recycle_action(self):
        """정렬 성공 후, 안전한 속도로 4초 동안 직진 (non-blocking)"""
        self.get_logger().info("🚀 정렬 성공! 이제 느린 속도로 4초간 직진을 시작합니다.")

        move_msg = Twist()
        move_msg.linear.x = 0.04  # 시각적으로 아주 조심스럽고 느리게 전진
        move_msg.angular.z = 0.0

        start_time = self.get_clock().now()
        duration = rclpy.duration.Duration(seconds=4.0)

        # 20Hz 주기로 시간 체크하며 속도 명령 발행 (await로 executor에 양보)
        while (self.get_clock().now() - start_time) < duration:
            self.cmd_vel_pub.publish(move_msg)
            await self._sleep(0.05)

        # 4초 이동 완료 후 최종 정지
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info("🎉 4초 직진 후 안전하게 정지 완료!")

        # 다음 정렬을 위해 플래그 초기화
        self.align_completed = False
        self.action_finished = True


def main(args=None):
    rclpy.init(args=args)
    node = RecycleTrackingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()