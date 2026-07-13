from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 첫 번째 노드: detected_object_id.py (토픽 발행용)
        Node(
            package='my_yolo_cpp_pkg',  # 실제 패키지 이름으로 확인
            executable='return_object_id_node', # setup.py left name
            name='object_detection_node',
            output='screen'
        ),
        # 두 번째 노드: best_yolo_node.py (시각화/디버깅용)
        Node(
            package='my_yolo_cpp_pkg',
            executable='best_yolo_node',
            name='yolo_visualization_node',
            output='screen'
        )
    ])
