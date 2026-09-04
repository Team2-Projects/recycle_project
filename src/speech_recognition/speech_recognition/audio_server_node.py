#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from my_interfaces.srv import AudioSpeech

# OpenVINO 모델 및 파이프라인 로드용
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
# from .intent_parser import IntentParser

class AudioServerNode(Node):
    def __init__(self):
        super().__init__('audio_server_node')

        # 1. 퍼블리셔 및 서비스 서버 생성
        self.publisher = self.create_publisher(String, 'speech_to_text', 10)
        self.srv = self.create_service(AudioSpeech, 'transcribe_audio', self.transcribe_callback)

        # 마지막으로 처리한 오디오 파일의 수정 시간(mtime) 기록용
        self.last_mtime = None

        self.get_logger().info("Whisper OpenVINO 모델 로딩 중...")


        # 로컬 PC 내 모델 저장 경로
        model_dir = os.path.expanduser('~/turtlebot3_ws/src/speech_recognition/models/whisper_tiny_openvino')

        try:
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir)
            self.processor = AutoProcessor.from_pretrained(model_dir)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device="cpu"
            )
            self.get_logger().info("Whisper OpenVINO 모델 로딩 완료. 서비스 준비됨.")
        except Exception as e:
            self.get_logger().error(f"모델 로딩 실패: {e}")

    def transcribe_callback(self, request, response):
        save_flag = request.save_flag
        self.get_logger().info(f"요청 수신 - save_flag: {save_flag}")

        if save_flag:
            try:
                # [강제 고정] 라즈베리 파이가 어디에 저장하든, 로컬 PC는 마운트 폴더 안의 파일을 직접 지정해서 읽음
                # 앞서 만들었던 SSHFS 마운트 폴더가 ~/rpi_share 라고 가정할 때의 경로입니다.
                local_file_path = os.path.expanduser('~/rpi_share/sound_files/recorded_audio.wav')

                self.get_logger().info(f"접근할 로컬 파일 경로: {local_file_path}")

                if not os.path.exists(local_file_path):
                    self.get_logger().warn(f"파일을 찾을 수 없습니다: {local_file_path}. (SSHFS 마운트 상태를 확인하세요)")
                    response.success = False
                    response.text = ""
                    return response

                # 1. 파일의 마지막 수정 시간 확인
                current_mtime = os.path.getmtime(local_file_path)

                # 2. 이전에 처리한 파일과 수정 시간이 완벽히 동일하다면 중복 요청 스킵
                if self.last_mtime is not None and current_mtime == self.last_mtime:
                    self.get_logger().warn("동일한 오디오 파일이 재요청되었습니다. (추론 스킵)")
                    response.success = True
                    response.text = ""
                    return response

                self.last_mtime = current_mtime

                # 3. OpenVINO 파이프라인을 통한 STT 추론
                result = self.pipe(
                    local_file_path,
                    generate_kwargs={
                        "language": "korean",
                        "task": "transcribe"
                    }
                )

                extracted_text = result['text'].strip()
                self.get_logger().info(f"인식된 텍스트: '{extracted_text}'")

                # 빈 텍스트가 아닐 때만 Topic 발행 및 응답
                if extracted_text:
                    msg = String()
                    msg.data = extracted_text
                    self.publisher.publish(msg)

                response.success = True
                response.text = extracted_text

            except Exception as e:
                self.get_logger().error(f"STT 변환 중 오류 발생: {e}")
                response.success = False
                response.text = ""
        else:
            response.success = False
            response.text = ""

        return response


def main(args=None):
    rclpy.init(args=args)
    node = AudioServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()