"""Keep the original launch entry point; optionally start tracking alone."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nodes(context):
    params = LaunchConfiguration('tracking_params').perform(context)
    tracking = Node(
        package='navigation', executable='recycle_tracking_node',
        name='recycle_tracking_node', parameters=[params], output='screen',
    )
    if LaunchConfiguration('tracking_only').perform(context) == 'true':
        # No AutoNav, Nav2, coverage, servo, or pan/tilt commands in this mode.
        return [tracking]
    return [
        Node(package='navigation', executable='coverage_node', name='coverage_node'),
        Node(package='navigation', executable='recycle', name='recycle'),
        tracking,
        Node(package='navigation', executable='auto_nav', name='auto_nav',
             parameters=[params], on_exit=Shutdown()),
    ]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('navigation'), 'config', 'recycle_tracking.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('tracking_params', default_value=default_params),
        DeclareLaunchArgument(
            'tracking_only', default_value='false', choices=['true', 'false'],
            description='true: tracking action only; false: original patrol/collection nodes',
        ),
        OpaqueFunction(function=_launch_nodes),
    ])
