from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node

def generate_launch_description():

    coverage = Node(
        package='navigation',
        executable='coverage_node',
        name='coverage_node'
    )

    recycle = Node(
        package='navigation',
        executable='recycle',
        name='recycle'
    )

    recycle_tracking_node = Node(
        package='navigation',
        executable='recycle_tracking_node',
        name='recycle_tracking_node'
    )

    auto_nav = Node(
        package='navigation',
        executable='auto_nav',
        name='auto_nav',
        on_exit=Shutdown()
    )

    return LaunchDescription([
        coverage,
        recycle,
        recycle_tracking_node,
        auto_nav
    ])