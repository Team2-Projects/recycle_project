import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pending_detection_age = DeclareLaunchArgument(
        'pending_detection_max_age_sec', default_value='2.0',
        description='Maximum age of a detection queued while navigation accepts a goal')
    # 1. 패키지 경로 설정 (nav2_bringup을 찾기 위함)
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # 2. Nav2 및 Gazebo 환경 포함 (기본 bringup 런치 파일 호출)
    # 실제 환경에 맞게 launch 파일 경로를 수정해야 할 수 있습니다 (예: tb3_simulation_launch.py 등)

    map_yaml_file = os.path.expanduser('~/turtlebot3_ws/map/map2.yaml')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'tb3_simulation_launch.py') 
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'map': map_yaml_file  # 여기가 추가된 부분입니다!
        }.items()
    )

    # 3. 사용자 커스텀 노드들 정의
    coverage = Node(package='navigation', executable='coverage_node', name='coverage_node')
    recycle = Node(package='navigation', executable='recycle', name='recycle')
    recycle_tracking = Node(package='navigation', executable='recycle_tracking_node', name='recycle_tracking_node')
    auto_nav = Node(
        package='navigation', executable='auto_nav', name='auto_nav',
        parameters=[{'pending_detection_max_age_sec': ParameterValue(
            LaunchConfiguration('pending_detection_max_age_sec'), value_type=float)}],
        on_exit=Shutdown())

    return LaunchDescription([
        pending_detection_age,
        nav2_launch,
        coverage,
        recycle,
        recycle_tracking,
        auto_nav
    ])
