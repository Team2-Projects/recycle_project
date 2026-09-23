"""추론 노드와 결과 영상 뷰어를 함께 실행한다."""

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    """모델은 검출 노드에서만 실행하고 로컬 창은 같은 결과를 표시한다."""
    return LaunchDescription([
        # 분류·YOLO 추론 및 제어 정보·표시 영상 발행
        Node(
            package='my_yolo_cpp_pkg',
            executable='classified_object_info_node',
            name='object_detection_node',
            output='screen'
        ),
        # 별도 명령 없이 결과 영상 창도 함께 연다. 여기서는 추론하지 않는다.
        Node(
            package='my_yolo_cpp_pkg',
            executable='best_yolo_node',
            name='yolo_visualization_node',
            output='screen'
        )
    ])
