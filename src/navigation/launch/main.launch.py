import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, Shutdown, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('navigation'), 'config', 'recycle_tracking.yaml',
    )
    params = LaunchConfiguration('tracking_params')
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
    recycle_tracking = Node(package='navigation', executable='recycle_tracking_node', name='recycle_tracking_node', parameters=[params], output='screen')
    auto_nav = Node(package='navigation', executable='auto_nav', name='auto_nav', parameters=[params], on_exit=Shutdown())

    return LaunchDescription([
        DeclareLaunchArgument('tracking_params', default_value=default_params),
        nav2_launch,
        coverage,
        recycle,
        recycle_tracking,
        auto_nav
    ])