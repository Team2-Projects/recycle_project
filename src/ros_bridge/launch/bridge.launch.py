from launch import LaunchDescription
from launch.actions import EmitEvent
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():

    ros_spring_bridge = Node(
        package='ros_bridge',
        executable='ros_spring_bridge',
        name='ros_spring_bridge',
        output='screen',
        on_exit=[
            EmitEvent(
                event=Shutdown(
                    reason='SpringBridge 종료'
                )
            )
        ]
    )

    ros_command_bridge = Node(
        package='ros_bridge',
        executable='ros_command_bridge',
        name='ros_command_bridge',
        output='screen',
        on_exit=[
            EmitEvent(
                event=Shutdown(
                    reason='CommandBridge 종료'
                )
            )
        ]
    )

    return LaunchDescription([
        ros_spring_bridge,
        ros_command_bridge
    ])