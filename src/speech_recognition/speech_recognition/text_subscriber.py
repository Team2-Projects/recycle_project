import os
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from geometry_msgs.msg import Twist
from .intent_parser import IntentParser
import json

from std_msgs.msg import Int32MultiArray
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    ReliabilityPolicy
)

class TtsSubscriber(Node):
    def __init__(self):
        super().__init__('tts_subscriber')
        
        # 1. 저장용 폴더 자동 생성 (없으면 생성)
        self.save_dir = 'sound_files'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 구독자 생성
        self.subscription = self.create_subscription(
            String,
            'speech_to_text',
            self.listener_callback,
            10
        )

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.path_pub = self.create_publisher(Int32MultiArray, '/patrol_indexs', qos)
        self.speech_pub = self.create_publisher(String, '/speech_pub', 10)
        
        # [추가] 라즈베리파이 스피커 노드로 음성 출력을 요청할 퍼블리셔 생성
        self.play_tts_pub = self.create_publisher(String, '/play_tts', 10)

        self.parser = IntentParser(self) # self 전달!
        self.command_flag = 0
      
        self.get_logger().info('🔊 자동 음성 출력 노드가 준비되었습니다.')

    def odom_callback(self, msg):
        # /odom에서 현재 선속도(linear.x)를 실시간으로 업데이트
        self.current_linear_x = msg.twist.twist.linear.x

    def is_auto_nav_alive(self):
        node_names = self.get_node_names()
        return any(
            name.strip('/') == 'auto_nav'
            for name in node_names
        )

    def listener_callback(self, msg):
        text = msg.data.strip()

        # 1. 빈 문자열이면 종료
        if not text:
            return

        if len(text) > 35:
            self.get_logger().warn(
                f'노이즈 후보 무시: "{text}" ({len(text)}자)'
            )
            return

        elif 0 < len(text) <= 35:
            self.command_flag = self.parser.return_flag(text)

        if self.command_flag == 0:
            patrol_paths = self.parser.get_patrol_indexs(text)
            self.get_logger().info(f'🔊 Patrol_paths for flag_0 수신: "{patrol_paths}"')

            speech_msg = String()
            speech_msg.data = json.dumps({
                "command": text,
                "commandIdx": int(self.command_flag),
                "patrolPaths": patrol_paths
            })
            self.speech_pub.publish(speech_msg)

            if self.is_auto_nav_alive():
                self.get_logger().warn(
                    "auto_nav already running"
                )
                return

            path_msg = Int32MultiArray()
            path_msg.data = patrol_paths
            self.path_pub.publish(path_msg)

            subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "navigation",
                    "navigation.launch.py"
                ],
                start_new_session=True
            )
            self.get_logger().info(
                "Launch started"
            )

        elif self.command_flag == 1:
            start_time = self.parser.get_start_time(text)
            self.get_logger().info(f'🔊 start_time for flag_1 수신: "{start_time}"')

            speech_msg = String()
            speech_msg.data = json.dumps({
                "command": text,
                "commandIdx": int(self.command_flag),
                "startTime": f"{int(start_time):02d}:00"
            })
            self.speech_pub.publish(speech_msg)

        elif self.command_flag == 2:
            return_text = self.parser.parse()
            self.get_logger().info(f'🔊 return_text for flag_2 수신: "{return_text}"')
            
            try:
                # 라즈베리파이의 speaker_node가 들을 수 있도록 /play_tts 토픽으로 텍스트 발행
                tts_msg = String()
                tts_msg.data = return_text 
                self.play_tts_pub.publish(tts_msg)
                
                self.get_logger().info(f'🔊 라즈베리파이로 음성 출력 요청 전송: "{return_text}"')

            except Exception as e:
                self.get_logger().error(f'음성 출력 요청 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TtsSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()