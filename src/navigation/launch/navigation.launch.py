"""Keep the original launch entry point; optionally start tracking alone."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nodes(context):
    params = LaunchConfiguration('tracking_params').perform(context)
    sim = LaunchConfiguration('use_sim_time').perform(context) == 'true'
    clock_params = {'use_sim_time': sim}
    tracking = Node(
        package='navigation', executable='recycle_tracking_node',
        name='recycle_tracking_node', parameters=[params, clock_params], output='screen',
    )
    collision = Node(
        package='nav2_collision_monitor', executable='collision_monitor',
        name='tracking_collision_monitor', parameters=[params, clock_params], output='screen',
    )
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_tracking_collision',
        parameters=[params, clock_params], output='screen',
    )
    if LaunchConfiguration('tracking_only').perform(context) == 'true':
        # No AutoNav, Nav2, coverage, servo, or pan/tilt commands in this mode.
        return [collision, lifecycle, tracking]
    return [
        collision, lifecycle,
        Node(package='navigation', executable='coverage_node', name='coverage_node'),
        Node(package='navigation', executable='recycle', name='recycle'),
        tracking,
        Node(package='navigation', executable='auto_nav', name='auto_nav',
             parameters=[params, clock_params], on_exit=Shutdown()),
    ]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('navigation'), 'config', 'recycle_tracking.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('tracking_params', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument(
            'tracking_only', default_value='false', choices=['true', 'false'],
            description='true: tracking action only; false: original patrol/collection nodes',
        ),
        OpaqueFunction(function=_launch_nodes),
    ])
