from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node

def generate_launch_description():

    ros_command_bridge = Node(
        package='ros_bridge',
        executable='ros_command_bridge',
        name='ros_command_bridge'
    )

    ros_spring_bridge = Node(
        package='ros_bridge',
        executable='ros_spring_bridge',
        name='ros_spring_bridge'
    )

    return LaunchDescription([
        ros_spring_bridge,
        ros_command_bridge
    ])