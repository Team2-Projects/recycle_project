from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node

def generate_launch_description():

    coverage = Node(
        package='navigation',
        executable='coverage_node',
        name='coverage_node'
    )

    object_detector = Node(
        package='navigation',
        executable='object_detector',
        name='object_detector'
    )

    recycle = Node(
        package='navigation',
        executable='recycle',
        name='recycle'
    )

    auto_nav = Node(
        package='navigation',
        executable='auto_nav',
        name='auto_nav',
        on_exit=Shutdown()
    )

    return LaunchDescription([
        coverage,
        object_detector,
        recycle,
        auto_nav
    ])