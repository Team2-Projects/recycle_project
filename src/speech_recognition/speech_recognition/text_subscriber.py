import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS
import pygame
from geometry_msgs.msg import Twist

class TtsSubscriber(Node):
    def __init__(self):
        super().__init__('tts_subscriber')
        
        # pygame 오디오 믹서 초기화
        pygame.mixer.init()
        
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

        # 오도메트리 구독자 생성 (현재 속도 측정용)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # 퍼블리셔 생성 (속도 제어용)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.collected_count = 0
        
        
        self.get_logger().info('🔊 자동 음성 출력 노드가 준비되었습니다.')

    def odom_callback(self, msg):
        # /odom에서 현재 선속도(linear.x)를 실시간으로 업데이트
        self.current_linear_x = msg.twist.twist.linear.x

    def listener_callback(self, msg):
        text = msg.data.strip()

        
        # 1. 빈 문자열이면 종료
        if not text:
            return

        # 2. 텍스트 수신 로그 출력
        self.get_logger().info(f'🔊 텍스트 수신: "{text}"')

       
        file_path = os.path.join(self.save_dir, f'speech.mp3')
        self.collected_count += 1

        # 3. 음성 명령에 따른 속도 제어 로직
        twist_msg = Twist()

        if "빠르게" in text:
            # 현재 속도가 0일 경우 기본 속도 0.1로 설정, 아니면 현재 속도의 2배
            base_speed = self.current_linear_x if self.current_linear_x != 0.0 else 0.05
            twist_msg.linear.x = base_speed * 2.0
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info(f'🚀 빠르게 가속: {twist_msg.linear.x:.2f} m/s')

        elif "느리게" in text:
            # 현재 속도가 0일 경우 기본 속도 0.2로 설정, 아니면 현재 속도의 0.5배
            base_speed = self.current_linear_x if self.current_linear_x != 0.0 else 0.05
            twist_msg.linear.x = base_speed * 0.5
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info(f'🐢 느리게 감속: {twist_msg.linear.x:.2f} m/s')

        elif "정지" in text:
            # 기본값 0.0인 Twist 메시지 발행 (정지)
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('🛑 로봇 정지')

        try:


            # 1. gTTS 변환 및 저장 (잘못된 logger 줄 삭제)
            tts = gTTS(text=text, lang='ko')
            tts.save(file_path)

            # 2. 음성 파일 재생
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            # 재생이 끝날 때까지 대기
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # 3. 언로드 및 성공 로그
            pygame.mixer.music.unload()
            self.get_logger().info('🔊 음성 출력 완료!')

        except Exception as e:
            self.get_logger().error(f'음성 출력 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TtsSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.mixer.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()