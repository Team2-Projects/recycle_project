# Recycle Project (쓰레기 분리수거 프로젝트) 
- 자율 순찰 및 재활용품 수거 프로젝트 

## Project 동작 흐름

```text
Coverage Path 생성 (Waypoint 생성)
        ↓
순차적으로 경로 순찰
        ↓
YOLO 객체 탐지
        ↓
객체 Tracking 및 정렬
        ↓
객체 접근
        ↓
출발지(분리수거장) 이동
        ↓
수거 후 경로 복귀 (후진, 회전)
        ↓
이전 순찰 위치로 이동
        ↓
남은 경로 순찰
```

## 특징 
- Coverage Path 기반 순찰 waypoint 생성 (X/Y/XY 최원점, Center)
- Nav2 기반 자율 주행
- YOLO 객체 탐지
- 객체 Tracking
- 카메라 중심 기반 객체 정렬
- Action 기반 재활용 기능 수행
- 수거 완료 후 순찰 재개 

## 개발 환경 
- OS: Ubuntu 22.04 / VirtualBox 가상환경 사용
 - ROS: ROS2 Humble 
- Robot: Turtlebot3 Waffle Pi
- Language: Python3 
- Simulator: Gazebo 
- Visualization: RViz2 

## Dependencies 
- 페이지 참고: https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup 
- turtlebot3에서 제공하는 Navigation2를 사용
- turtlebot3패키비 설치 후 빌드하여 사용
```bash
# 기본 turtlebot 패키지
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git

# 시뮬레이션 gazebo
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

# YOLO사용
pip install ultralytics

# Open CV 사용
pip install opencv-python
```

## Build
```bash
cd ~/turtlebot3_ws
colcon build --symlink-install
source install/setup.bash
```

## Requirements 
- Ubuntu 22.04 
- ROS2 Humble 
- TurtleBot3 Navigation2
- Gazebo 
- RViz2 
- Python 3 
- Ultralytics YOLOv8 
- OpenCV
