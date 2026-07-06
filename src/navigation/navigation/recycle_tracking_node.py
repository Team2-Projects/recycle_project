import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from navigation_interface.action import RecycleActionMsg
import asyncio  # 여전히 필요합니다 (Future 사용)

class RecycleTrackingNode(Node):

    def __init__(self):
        super().__init__('recycle_tracking_node')
        self.cb_group = ReentrantCallbackGroup()
        self.latest_object = None

        self.sub = self.create_subscription(
            DetectedObject, '/detected_object_info', self.obj_callback, 10, callback_group=self.cb_group)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tracking_cli = self.create_client(
            SetTracking, 'set_tracking_mode', callback_group=self.cb_group)

        while not self.tracking_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Tracking Service 기다리는 중...")

        self._action_server = ActionServer(
            self,
            RecycleActionMsg,
            "recycle_tracking_action",
            execute_callback=self.execute_callback,
            callback_group=self.cb_group
        )

    # [핵심 수정] ROS 2 환경에서 안전한 비동기 sleep 함수
    async def ros_sleep(self, duration):
        future = asyncio.Future()
        timer = self.create_timer(duration, lambda: future.set_result(None) or timer.cancel())
        await future

    def obj_callback(self, msg):
        self.latest_object = msg

    async def call_tracking_srv(self, enable):
        req = SetTracking.Request()
        req.enable = enable
        try:
            response = await self.tracking_cli.call_async(req)
            return response.success
        except Exception as e:
            self.get_logger().error(f"서비스 호출 예외: {e}")
            return False

    async def execute_callback(self, goal_handle):
        self.get_logger().info("접근 및 추적 시작 요청")
        if not await self.call_tracking_srv(True):
            goal_handle.abort()
            return RecycleActionMsg.Result()
        
        align_success = await self.align_robot()
        approach_success = await self.approach_robot(goal_handle) if align_success else False
        
        await self.call_tracking_srv(False)
        
        if align_success and approach_success:
            goal_handle.succeed()
            return RecycleActionMsg.Result()
        else:
            goal_handle.abort()
            return RecycleActionMsg.Result()

    async def align_robot(self):
        self.get_logger().info("물체 정렬 루프 시작...")
        while True:
            if self.latest_object is None or self.latest_object.id == -1:
                await self.ros_sleep(0.05)
                continue

            diff = 320 - self.latest_object.coord[0]
            if abs(diff) < 10:
                break

            msg = Twist()
            msg.angular.z = (1 if diff > 0 else -1) * 0.1
            self.cmd_vel_pub.publish(msg)
            await self.ros_sleep(0.05)

        self.cmd_vel_pub.publish(Twist())
        return True

    async def approach_robot(self, goal_handle):
        velocity = 0.10
        probe_duration = 0.5
        while self.latest_object is None:
            await self.ros_sleep(0.1)

        h1 = self.latest_object.coord[3]
        total_move_time = 0.0

        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False

            msg = Twist()
            msg.linear.x = velocity
            self.cmd_vel_pub.publish(msg)
            await self.ros_sleep(probe_duration)
            self.cmd_vel_pub.publish(Twist())
            total_move_time += probe_duration

            if self.latest_object is None: continue

            h_current = self.latest_object.coord[3]
            diff = h_current - h1
            if diff >= 1.0: break
            if total_move_time >= 3.0: return False

       
        
        Z = (velocity * total_move_time) * (h_current / diff) - (velocity * total_move_time)
        remaining = max(0.0, Z - 0.05)
        
        msg.linear.x = velocity
        self.cmd_vel_pub.publish(msg)
        await self.ros_sleep(remaining / velocity)
        
        self.cmd_vel_pub.publish(Twist())
        return True

def main(args=None):
    rclpy.init(args=args)
    node = RecycleTrackingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()