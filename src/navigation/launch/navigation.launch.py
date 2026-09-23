from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    pending_detection_age = DeclareLaunchArgument(
        'pending_detection_max_age_sec', default_value='2.0',
        description='Maximum age of a detection queued while navigation accepts a goal')

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
        parameters=[{'pending_detection_max_age_sec': ParameterValue(
            LaunchConfiguration('pending_detection_max_age_sec'), value_type=float)}],
        on_exit=Shutdown()
    )

    return LaunchDescription([
        pending_detection_age,
        coverage,
        recycle,
        recycle_tracking_node,
        auto_nav
    ])
