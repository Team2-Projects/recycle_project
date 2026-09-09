"""Keep the original YOLO launch name and select exactly one perception path.

Default: the tested YOLO-only path (no pre-classifier load or inference).
Opt-in: the untouched legacy classifier path, once its model/paths are ready.
Do not start classified_object_info_node or best_yolo_node separately as well.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
    if LaunchConfiguration('use_preclassifier').perform(context) == 'true':
        # Preserve the original executable and source file. Its model paths
        # and branch logic are controlled by the team, NOT by yolo_only.yaml.
        return [Node(
            package='my_yolo_cpp_pkg', executable='classified_object_info_node',
            name='object_detection_node', output='screen',
        )]
    overrides = {}
    for name in ('model_path', 'image_topic', 'target_class_id'):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = int(value) if name == 'target_class_id' else value
    return [Node(
        package='my_yolo_cpp_pkg', executable='yolo_only_info_node',
        name='yolo_only_node', output='screen',
        parameters=[LaunchConfiguration('vision_params').perform(context), overrides],
    )]


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('my_yolo_cpp_pkg'), 'config', 'yolo_only.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('vision_params', default_value=cfg),
        DeclareLaunchArgument(
            'use_preclassifier', default_value='false', choices=['true', 'false'],
            description='true: original classifier path (model/paths must be ready)',
        ),
        DeclareLaunchArgument('model_path', default_value='', description='YOLO-only override'),
        DeclareLaunchArgument('image_topic', default_value='', description='YOLO-only override'),
        DeclareLaunchArgument('target_class_id', default_value='', description='YOLO-only override'),
        OpaqueFunction(function=_launch_node),
    ])
