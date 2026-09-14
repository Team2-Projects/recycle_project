import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    # Keep the existing simulation entry point. Use the SAME tracking safety
    # launch as the real robot so this path cannot bypass Collision Monitor.
    project_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('navigation'), 'launch', 'navigation.launch.py')),
        launch_arguments={'tracking_params': params, 'use_sim_time': 'true'}.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument('tracking_params', default_value=default_params),
        nav2_launch,
        project_nodes,
    ])
