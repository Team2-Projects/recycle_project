import rclpy
import asyncio
from rclpy.node import Node
from geometry_msgs.msg import Twist
from my_yolo_msgs.msg import DetectedObject

class RecycleTrackingNode(Node):
    def __init__(self):
        super().__init__('recycle_tracking_node')

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
        
        # YOLO 노드가 보내주는 토픽 구독
        self.yolo_sub = self.create_subscription(
            DetectedObject, '/detected_object_info', self.yolo_callback, 10)

        self.get_logger().info("♻️ 실시간 분리수거 정렬 및 4초 직진 노드가 시작되었습니다.")

    async def execute_callback(self, goal_handle):
        self.action_finished = False
        self.align_completed = False

        while not self.action_finished:
            await asyncio.sleep(0.05)
        goal_handle.succeed()
        
        result = RecycleActionMsg.Result()
        result.success = True
        return result

    def yolo_callback(self, msg):
        # 정렬이 이미 완전히 끝난 상태라면 YOLO 콜백에서 로봇을 제어하지 않음
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
                self.get_logger().info(f"🔍 물체 왼쪽 감지 (X: {x:.1f}) -> 왼쪽으로 회전 중...", throttle_duration_sec=0.5)
                self.cmd_vel_pub.publish(msg_cmd)
            
            # [규칙 2] X값이 400보다 크면 물체가 오른쪽에 있으므로 오른쪽으로 회전
            elif x > 350.0:
                msg_cmd.angular.z = -0.2
                self.get_logger().info(f"🔍 물체 오른쪽 감지 (X: {x:.1f}) -> 오른쪽으로 회전 중...", throttle_duration_sec=0.5)
                self.cmd_vel_pub.publish(msg_cmd)
            
            # [규칙 3] X값이 300 ~ 400 사이에 들어오면 정중앙에 위치한 것이므로 멈춤!
            else:
                msg_cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(msg_cmd)
                self.get_logger().info(f"🎯 정렬 완료! 물체가 중앙에 있습니다. (X: {x:.1f})")
                
                # 정렬 플래그를 참으로 만들고 다음 액션 실행
                self.align_completed = True
                self.launch_recycle_action()

        # 2. 물체가 인식되지 않았다면 안전을 위해 로봇을 멈춤
        else:
            msg_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(msg_cmd)

    def launch_recycle_action(self):
        """ 🔴 [수정] 정렬 성공 후, 안전한 속도로 4초 동안 직진 """
        self.get_logger().info("🚀 정렬 성공! 이제 느린 속도로 4초간 직진을 시작합니다.")
        
        # 주행 속도 설정 (0.04 m/s = 시각적으로 아주 조심스럽고 느리게 전진)
        move_msg = Twist()
        move_msg.linear.x = 0.04  
        move_msg.angular.z = 0.0

        # ROS 2 시간 측정 시작점 설정
        start_time = self.get_clock().now()
        duration = rclpy.duration.Duration(seconds=4.0) # 4초 타겟 설정

        # 20Hz 주기로 시간 체크하며 속도 명령 발행
        rate = self.create_rate(20)
        while rclpy.ok():
            # 현재 흐른 시간 계산
            elapsed_time = self.get_clock().now() - start_time
            
            # 4초가 지나면 루프 탈출
            if elapsed_time >= duration:
                break
                
            self.cmd_vel_pub.publish(move_msg)
            rate.sleep()

        # 4초 이동 완료 후 최종 정지
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info("🎉 4초 직진 후 안전하게 정지 완료!")

        # 다음 정렬을 위해 플래그 초기화 (무한 반복)
        self.get_logger().info("🔄 다음 정렬을 위해 시스템을 초기화합니다. 다시 물체를 탐지합니다...")
        self.align_completed = False

        self.action_finished = True


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = RecycleTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()