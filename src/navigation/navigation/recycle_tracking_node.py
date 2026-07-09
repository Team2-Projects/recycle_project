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
import math

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

        # self.get_logger().info(f"YOLO 추적 모드 변경 요청 보냄: {enable}")
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

        # self.get_logger().info(f"YOLO 추적 모드 변경 완료 응답 수신: {future.result().success}")
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
    #     # self.get_logger().info("물체 정렬 루프 시작...")
        
    #     # self.get_logger().info(f"target_x: {target_x:.2f}")
    #     diff = 320 - target_x
    #     diff_angle = abs(diff/10.3)

    #     while rclpy.ok():
    #         # self.get_logger().info(f"diff: {diff:.2f}")
    #         # self.get_logger().info(f"diff_angle: {diff_angle:.2f}")
    #         if abs(diff_angle) < 0.3:
    #             self.get_logger().info(f"정렬 성공! 오차 angle: {diff_angle:.2f}")
    #             break

    #         msg = Twist()
    #         msg.angular.z = (1 if diff > 0 else -1) * 0.02
            
    #         self.cmd_vel_pub.publish(msg)


    #         time.sleep(0.2)
            
    #         diff_angle -= abs(msg.angular.z * 0.2) * (180.0 / math.pi)
    #     self.cmd_vel_pub.publish(Twist())

    #     return True

    #     실시간 데이터를 사용하므로 conf를 0.2-3정도의 낮은 값으로 맞추세요.
    def align_robot(self, target_x): # target_w는 초기값일 뿐, 루프에선 쓰지 마세요
        self.get_logger().info("물체 정렬 루프 시작...")
    
        self.get_logger().info(f"target_x: {target_x:.2f}")
        while rclpy.ok():
            # 1. 실시간으로 최신 데이터 가져오기 (매우 중요!)
            if self.latest_object is None:
                init_diff = 320 - target_x
                msg = Twist()
                # 0.2 -> 0.05
                msg.angular.z = (1 if diff > 0 else -1) * 0.05
                self.cmd_vel_pub.publish(msg)
                    
                time.sleep(0.05)
                
                if abs(init_diff) < 20:
                    self.get_logger().info(f"정렬 성공! 오차 픽셀: {init_diff:.2f}")
                    break
                # self.cmd_vel_pub.publish(Twist())
                # time.sleep(0.05)
                # msg = Twist()
                # msg.angular.z = 0.00
                # msg.linear.x = 0.02
                # self.cmd_vel_pub.publish(msg)
                # time.sleep(0.05)
                # self.cmd_vel_pub.publish(Twist())
                # time.sleep(0.05)
                continue
            
            # 1. 실시간 중심점 계산
            current_x = self.latest_object.coord[0]
            target_x = current_x
            diff = 320 - current_x # 화면 중앙(320)과 현재 물체 위치의 차이
                # 3. 회전 명령 (오른쪽에 있으면 양수, 왼쪽에 있으면 음수)
            msg = Twist()
            # 0.2 -> 0.05
            msg.angular.z = (1 if diff > 0 else -1) * 0.05
            self.cmd_vel_pub.publish(msg)
                
            time.sleep(0.05) # 너무 자주 보내지 않게 잠시 대기

                # 4. 정렬 조건 (오차 20픽셀 이내)
            if abs(diff) < 20:
                self.get_logger().info(f"정렬 성공! 오차 픽셀: {diff:.2f}")
                break


        self.cmd_vel_pub.publish(Twist()) # 정렬 완료 시 정지

    
        return True

# 박스가 없어지는 문제. 
    def approach_robot(self, goal_handle):
        current_x = self.latest_object.coord[0]
        target_x = current_x
        diff = 320 - current_x # 화면 중앙(320)과 현재 물체 위치의 차이
                # 3. 회전 명령 (오른쪽에 있으면 양수, 왼쪽에 있으면 음수)

        if abs(diff) >= 20:
            self.align_robot(current_x)
        

        # 로봇이 직진하는 속력.
        velocity = 0.10
        # 이 시간동안 움직여라. 
        probe_duration = 1.0
        # while self.latest_object is None:
        #     self.get_logger().info("접근 전 YOLO 데이터 대기 중...", throttle_duration_sec=2.0)
        #     time.sleep(0.05)
        last_approach_time = 3.0
        while rclpy.ok():
            if self.latest_object is None:
                # 감지를 못할때는 정말 천천히 움직이면서 물체를 감지하도록 한다.
                slow_motion = 0.02
                msg = Twist()
                msg.linear.x = slow_motion
                self.cmd_vel_pub.publish(msg)
                time.sleep(0.2)
                self.cmd_vel_pub.publish(Twist())
                continue
            
            # current_h = self.latest_object.coord[3]

            current_h = self.latest_object.coord[3]
            current_y = self.latest_object.coord[1]
            # later_h = self.latest_object.coord[3]
            lower_y = current_y + (current_h/2)
            # self.get_logger().info("박스 위 y좌표 = {}".format(lower_y))            

            if lower_y >= 430:
                msg = Twist()
                msg.linear.x = velocity

                ## 아래 두 줄은 0.10(velocity)m/s로 probe_duration(1초)동안 움직여라.
                self.cmd_vel_pub.publish(msg)
                time.sleep(last_approach_time)
                break

            msg = Twist()
            msg.linear.x = velocity

                ## 아래 두 줄은 0.10(velocity)m/s로 probe_duration(1초)동안 움직여라.
            self.cmd_vel_pub.publish(msg)
            time.sleep(probe_duration)
                

            self.cmd_vel_pub.publish(Twist())

        self.get_logger().info("접근 완료")

        return True

            # h_current = self.latest_object.coord[3]

#     def approach_robot(self, goal_handle):

#         non_tracking_velocity = 0.03
#         probe_duration = 0.5
#         start_time = self.get_clock().now()

#         total_no_detect_time = 0
#         #slow_motion is 0.05 -> 0.01
#         slow_motion = 0.02
#         while self.latest_object is None:
#             st = self.get_clock().now()
#             self.get_logger().info("접근 전 YOLO 데이터 대기 중...", throttle_duration_sec=2.0)
#             msg = Twist()
#             msg.linear.x = slow_motion
#             self.cmd_vel_pub.publish(msg)
#             time.sleep(0.2)
#             self.cmd_vel_pub.publish(Twist())
#             et = self.get_clock().now()
#             total_no_detect_time += (et - st)
         
            
#         h1 = self.latest_object.coord[3]
        
#         total_move_time = 0.0
#         self.get_logger().info(f"접근 루프 시작 (초기 h1: {h1:.2f})")
        
#         # [수정] 시작 시간 기록을 루프 외부로 뺍니다.
#         while rclpy.ok():

#             msg = Twist()
#             msg.linear.x = non_tracking_velocity
#             self.cmd_vel_pub.publish(msg)

#             time.sleep(probe_duration)

#             # [수정] 누적 시간 대신 정확한 경과 시간을 계산
#             now = self.get_clock().now()
#             total_move_time = (now - start_time).nanoseconds * 1e-9
       

#             if self.latest_object is None:
#                 continue

#             h_current = self.latest_object.coord[3]

#             diff = h_current - h1

#             self.get_logger().info(
#                 f"접근 중: 현재 높이={h_current:.2f} 높이 차이={diff:.2f}"
#             )

# # 실제 코드 구현시 데이터 통신 및 잡음 문제로 인해, 이상적인 값이 안나올 수 있으니 일정 비율만큼만 고려한다.
#             if diff >= 0.10*h1:
#                 break

#             if total_move_time >= 10.0:

#                 self.get_logger().error(
#                     "{}초 동안 높이 변화 없음".format(total_move_time)
#                 )

#                 self.cmd_vel_pub.publish(Twist())

#                 return False
#         self.cmd_vel_pub.publish(Twist())

#         velocity = 0.1
#         d = non_tracking_velocity * total_move_time + slow_motion * total_no_detect_time

#         Z = d * (h_current / diff) 
#         Z = Z - (1.0*d)

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

    # def approach_robot(self, goal_handle):
    #     non_tracking_velocity = 0.03
    #     slow_motion = 0.02

    #     # 1. YOLO 탐지될 때까지 저속 접근
    #     while self.latest_object is None:
    #         self.get_logger().info("접근 전 YOLO 데이터 대기 중...", throttle_duration_sec=2.0)
    #         msg = Twist()
    #         msg.linear.x = slow_motion
    #         self.cmd_vel_pub.publish(msg)
    #         time.sleep(0.2)
    #         self.cmd_vel_pub.publish(Twist())

    #     h1 = self.latest_object.coord[3]
    #     self.get_logger().info(f"접근 루프 시작 (초기 h1: {h1:.2f})")

    #     # 트래킹 루프 전용 시작 시각 (여기서부터 이동 거리 m만 계산)
    #     track_start_time = self.get_clock().now()
    #     probe_duration = 0.5
    #     m = 0.0
    #     h_current = h1
    #     diff = 0.0

    #     # 2. 실제 접근 + 높이 변화 트래킹
    #     while rclpy.ok():
    #         msg = Twist()
    #         msg.linear.x = non_tracking_velocity
    #         self.cmd_vel_pub.publish(msg)
    #         time.sleep(probe_duration)

    #         now = self.get_clock().now()
    #         elapsed = (now - track_start_time).nanoseconds * 1e-9
    #         m = elapsed * non_tracking_velocity  # 트래킹 시작 이후 실제 이동 거리

    #         if self.latest_object is None:
    #             continue

    #         h_current = self.latest_object.coord[3]
    #         diff = h_current - h1

    #         self.get_logger().info(f"접근 중: 현재 높이={h_current:.2f} 높이 차이={diff:.2f}")

    #         if diff >= 0.20 * h1:
    #             break

    #         if elapsed >= 10.0:
    #             self.get_logger().error(f"{elapsed:.1f}초 동안 높이 변화 없음")
    #             self.cmd_vel_pub.publish(Twist())
    #             return False

    #     self.cmd_vel_pub.publish(Twist())

    #     # 3. 남은 거리 계산 (R지점 기준 거리 Z_R - 이미 이동한 m)
    #     remaining = m * h1 / diff
    #     remaining = max(0.0, remaining - 0.1)  # 안전 마진

    #     velocity = 0.1
    #     move_time = remaining / velocity
    #     self.get_logger().info(f"남은 거리 {remaining:.3f}m 만큼 {move_time:.2f}초간 최종 전진합니다.")

    #     msg = Twist()
    #     msg.linear.x = velocity
    #     self.cmd_vel_pub.publish(msg)
    #     time.sleep(move_time)
    #     self.cmd_vel_pub.publish(Twist())

    #     self.get_logger().info("접근 완료")
    #     return True


def main(args=None):
    rclpy.init(args=args)
    node = RecycleTrackingNode()
    executor = MultiThreadedExecutor(num_threads=3)
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