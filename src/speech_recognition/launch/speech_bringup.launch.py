from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 첫 번째 노드: detected_object_id.py (토픽 발행용)
        Node(
            package='speech_recognition',  # 실제 패키지 이름으로 확인
            executable='audio_server_node', # setup.py left name
            name='stt_node',
            output='screen'
        ),
        # 두 번째 노드: best_yolo_node.py (시각화/디버깅용)
        Node(
            package='speech_recognition',
            executable='tts_subscriber_node',
            name='tts_node',
            output='screen'
        )
    ])
