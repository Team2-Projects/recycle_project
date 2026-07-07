import time
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist

from my_yolo_msgs.msg import DetectedObject
from my_yolo_msgs.srv import SetTracking
from navigation_interface.action import RecycleActionMsg


class RecycleTrackingNode(Node):

    def __init__(self):
        super().__init__('recycle_tracking_node')

        self.cb_group = ReentrantCallbackGroup()

        self.latest_object = None
        self.target_h_threshold = 320

        self.sub = self.create_subscription(
            DetectedObject,
            '/detected_object_info',
            self.obj_callback,
            10,
            callback_group=self.cb_group)

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        self.tracking_cli = self.create_client(
            SetTracking,
            'set_tracking_mode',
            callback_group=self.cb_group)

        while not self.tracking_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Tracking Service 기다리는 중...")

        self._action_server = ActionServer(
            self,
            RecycleActionMsg,
            "recycle_tracking_action",
            execute_callback=self.execute_callback,
            callback_group=self.cb_group
        )

    def obj_callback(self, msg):
        self.latest_object = msg

    def call_tracking_srv(self, enable):
        req = SetTracking.Request()
        req.enable = enable

        self.get_logger().info(f"YOLO 추적 모드 변경 요청 보냄: {enable}")
        future = self.tracking_cli.call_async(req)

        event = Event()

        def done_callback(fut):
            event.set()

        future.add_done_callback(done_callback)

        while rclpy.ok():
            if event.wait(timeout=0.05):
                break

        if future.result() is None:
            self.get_logger().error("Tracking Service 호출 실패")
            return False

        self.get_logger().info(f"YOLO 추적 모드 변경 완료 응답 수신: {future.result().success}")
        return True

    def execute_callback(self, goal_handle):

        self.get_logger().info("Recycle Tracking 시작")

        if not self.call_tracking_srv(True):
            goal_handle.abort()
            result = RecycleActionMsg.Result()
            result.success = False
            result.message = 'YOLO 추적 서비스 호출 실패'
            return result

        self.get_logger().info("정렬(Align) 단계 진입")
        if not self.align_robot(goal_handle.request.target_x):
            self.call_tracking_srv(False)
            goal_handle.abort()
            result = RecycleActionMsg.Result()
            result.success = False
            result.message = '정렬 실패'
            return result

        self.get_logger().info("접근(Approach) 단계 진입")
        if not self.approach_robot(goal_handle):
            self.call_tracking_srv(False)
            goal_handle.abort()
            result = RecycleActionMsg.Result()
            result.success = False
            result.message = '접근 실패 (거리 변화 감지 안됨)'
            return result

        self.call_tracking_srv(False)

        goal_handle.succeed()

        self.get_logger().info("Recycle Tracking 완료")

        result = RecycleActionMsg.Result()
        result.success = True
        result.message = '정렬 및 접근 완료'
        return result


    # # def align_robot(self, target_x):
    # def align_robot(self, target_x):
    #     self.get_logger().info("물체 정렬 루프 시작...")
        
    #     self.get_logger().info(f"target_x: {target_x:.2f}")
    #     diff = 320 - target_x
    #     diff_angle = abs(diff/10.2)

    #     while rclpy.ok():
    #         if self.latest_object is None:
    #             self.get_logger().info("YOLO 토픽 데이터 대기 중...", throttle_duration_sec=2.0)
    #             time.sleep(0.05)
    #             continue

    #         # if self.latest_object.id == -1:
    #         #     self.get_logger().info("정렬 중: 감지된 물체가 없음 (id == -1)", throttle_duration_sec=2.0)
    #         #     time.sleep(0.05)
    #         #     continue    
    #         self.get_logger().info(f"diff: {diff:.2f}")
    #         self.get_logger().info(f"diff_angle: {diff_angle:.2f}")
    #         if abs(diff_angle) < 1:
    #             self.get_logger().info(f"정렬 성공! 오차 angle: {diff_angle:.2f}")
    #             break

    #         msg = Twist()
    #         msg.angular.z = (1 if diff > 0 else -1) * 0.2
            
    #         self.cmd_vel_pub.publish(msg)

    #         time.sleep(0.2)
    #         diff_angle -= abs(msg.angular.z * 0.2)
    #     self.cmd_vel_pub.publish(Twist())

    #     return True

        # 실시간 데이터를 사용하므로 conf를 0.2-3정도의 낮은 값으로 맞추세요.
    def align_robot(self, target_w): # target_w는 초기값일 뿐, 루프에선 쓰지 마세요
        self.get_logger().info("물체 정렬 루프 시작...")
    
        
        while rclpy.ok():
            # 1. 실시간으로 최신 데이터 가져오기 (매우 중요!)
            if self.latest_object is None:
                time.sleep(0.05)
                continue
            
            # 1. 실시간 중심점 계산
            current_x = self.latest_object.coord[0]
            diff = 320 - current_x # 화면 중앙(320)과 현재 물체 위치의 차이
                # 3. 회전 명령 (오른쪽에 있으면 양수, 왼쪽에 있으면 음수)
            msg = Twist()
            msg.angular.z = (1 if diff > 0 else -1) * 0.2
            self.cmd_vel_pub.publish(msg)
                
            time.sleep(0.05) # 너무 자주 보내지 않게 잠시 대기

                # 4. 정렬 조건 (오차 10픽셀 이내)
            if abs(diff) < 10:
                self.get_logger().info(f"정렬 성공! 오차 픽셀: {diff:.2f}")
                break


        self.cmd_vel_pub.publish(Twist()) # 정렬 완료 시 정지
        return True

    def approach_robot(self, goal_handle):
        velocity = 0.10
        probe_duration = 0.5
        # while self.latest_object is None:
        #     self.get_logger().info("접근 전 YOLO 데이터 대기 중...", throttle_duration_sec=2.0)
        #     time.sleep(0.05)

        while rclpy.ok():
            if self.latest_object is None:
                slow_motion = 0.01
                msg = Twist()
                msg.linear.x = velocity
                self.cmd_vel_pub.publish(msg)
                time.sleep(0.05)
                self.cmd_vel_pub.publish(Twist())
                continue
            
        
            current_h = self.latest_object.coord[3]

            msg = Twist()
            msg.linear.x = velocity
            self.cmd_vel_pub.publish(msg)
            self.cmd_vel_pub.publish(Twist())

            time.sleep(probe_duration)

            later_h = self.latest_object.coord[3]

            if later_h >= 300:
                break

        self.cmd_vel_pub.publish(Twist())

        self.get_logger().info("접근 완료")

        return True
            # h_current = self.latest_object.coord[3]

#     def approach_robot(self, goal_handle):
#         velocity = 0.10
#         probe_duration = 0.5

#         while self.latest_object is None:
#             self.get_logger().info("접근 전 YOLO 데이터 대기 중...", throttle_duration_sec=2.0)
#             time.sleep(0.05)

#         h1 = self.latest_object.coord[3]

#         total_move_time = 0.0
#         self.get_logger().info(f"접근 루프 시작 (초기 h1: {h1:.2f})")

#         while rclpy.ok():

#             msg = Twist()
#             msg.linear.x = velocity
#             self.cmd_vel_pub.publish(msg)

#             time.sleep(probe_duration)

#             # self.cmd_vel_pub.publish(Twist())

#             total_move_time += probe_duration

#             if self.latest_object is None:
#                 continue

#             h_current = self.latest_object.coord[3]

#             diff = h_current - h1

#             self.get_logger().info(
#                 f"접근 중: h={h_current:.2f} diff={diff:.2f}"
#             )

# # 실제 코드 구현시 데이터 통신 및 잡음 문제로 인해, 이상적인 값이 안나올 수 있으니 일정 비율만큼만 고려한다.
#             # if diff >= :
#             #     break

#             if total_move_time >= 3.0:

#                 self.get_logger().error(
#                     "3초 동안 거리 변화 없음"
#                 )

#                 self.cmd_vel_pub.publish(Twist())

#                 return False

#         d = velocity * total_move_time

#         Z = d * (h_current / diff) 
#         Z = Z - d

#         self.get_logger().info(
#             f"계산 거리 = {Z:.3f}"
#         )

#         remaining = max(0.0, Z - 0.05)

#         move_time = remaining / velocity
#         self.get_logger().info(f"남은 거리 {remaining:.3f}m 만큼 {move_time:.2f}초간 최종 전진합니다.")

#         msg = Twist()
#         msg.linear.x = velocity

#         self.cmd_vel_pub.publish(msg)

#         time.sleep(move_time)

#         self.cmd_vel_pub.publish(Twist())

#         self.get_logger().info("접근 완료")

#         return True


def main(args=None):

    rclpy.init(args=args)

    node = RecycleTrackingNode()

    executor = MultiThreadedExecutor()

    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("시그널 감지: 노드를 종료합니다.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()