import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS
import pygame
from geometry_msgs.msg import Twist
from .intent_parser import IntentParser

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

        self.collected_count = 0
        self.parser = IntentParser()
        self.command_flag = 0
      
        self.get_logger().info('🔊 자동 음성 출력 노드가 준비되었습니다.')

    def odom_callback(self, msg):
        # /odom에서 현재 선속도(linear.x)를 실시간으로 업데이트
        self.current_linear_x = msg.twist.twist.linear.x

    def listener_callback(self, msg):
        text = msg.data.strip()

        
        # 1. 빈 문자열이면 종료
        if not text:
            return


        self.command_flag = self.parser.return_flag(text)

        if self.command_flag == 0:
            patrol_paths = self.parser.get_patrol_indexs(text)
            self.get_logger().info(f'🔊 Patrol_paths for flag_0 수신: "{patrol_paths}"')

        elif self.command_flag == 1:
            start_time= self.parser.get_start_time(text)
            self.get_logger().info(f'🔊 start_time for flag_1 수신: "{start_time}"')

        elif self.command_flag == 2:
            return_text = self.parser.parse()
            self.get_logger().info(f'🔊 return_text for flag_2 수신: "{return_text}"')


       

       
            file_path = os.path.join(self.save_dir, f'return_text.mp3')
            self.collected_count += 1

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