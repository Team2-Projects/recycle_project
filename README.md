# Recycle Project (쓰레기 분리수거 프로젝트)

TurtleBot3 Waffle Pi와 ROS2 Humble을 활용한 자율 순찰·재활용품 수거 프로젝트입니다.
물체를 탐지·수거하고 종류별 분리수거장에서 하역한 뒤 순찰을 재개합니다.

## Project 동작 흐름

```text
웨이포인트 경로 생성 → 순찰 → 객체 탐지 → 추적·정렬·접근 → 수거
  ├─ 적재 공간 있음 → 기존 순찰 목표로 복귀
  └─ 적재 공간 부족 → 분리수거장 이동·하역 → HOME → 경로 처음부터 순찰

순찰 경로 완료 → HOME → 적재물이 있으면 하역, 없으면 순찰 종료
```

## 특징

- 지정 좌표 기반 Coverage Path 생성 및 Nav2·TEB 자율주행
- 캔·종이·플라스틱·일반쓰레기 분류와 카메라 중심 기반 추적·접근
- ROS2 Action과 서보 제어를 통한 수거·하역 및 같은 종류의 추가 수거
- 첫 물체의 접근 중 검출을 종합해 분류하고, 애매할 때만 수거함 영상으로 재확인
- 한 번의 YOLO 추론 결과를 주행 제어와 영상 표시에 공유하며, 실행 시 영상 창도 자동으로 표시
- 분리수거장 이동과 HOME 복귀 중 추론 중지, 순찰 재개 시 추론 재시작

## 개발 환경

| 구분 | 구성 |
| --- | --- |
| OS | Ubuntu 22.04 / PC는 VirtualBox 가상환경 사용 |
| Robot | TurtleBot3 Waffle Pi, Raspberry Pi 4 Model B |
| ROS·주행 | ROS2 Humble, Nav2, TEB |
| Language·Vision | Python 3, Ultralytics YOLOv8, OpenVINO, OpenCV |
| 시각화·시뮬레이션 | RViz2, Gazebo(시뮬레이션용) |

## Dependencies

기본 ROS·로봇 패키지는 [TurtleBot3 공식 설치 안내](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup)의 Humble 항목을 참고합니다.

- **PC:** TurtleBot3 Navigation2, ROS2용 `teb_local_planner`, `cv_bridge`, `vision_msgs`
- **로봇:** `turtlebot3_bringup`, `v4l2_camera`, `compressed_image_transport`, 별도 `control_motor` 패키지
- **Python:** Ultralytics, OpenVINO, OpenCV, NumPy, Pillow, SciPy

`control_motor`와 TEB 플러그인은 이 저장소에 포함되어 있지 않습니다.
현재 로봇의 서보 설정은 `pigpiod` 서비스 자동 실행을 전제로 합니다.
분류·탐지 모델은 [models](src/my_yolo_cpp_pkg/models)에 포함되어 패키지 빌드 시 함께 설치됩니다.

## 빌드 및 실행

아래 명령은 저장소가 `~/turtlebot3_ws/recycle_project`에 있는 환경 기준입니다.

### PC 패키지 빌드

사용자 정의 메시지·서비스 패키지까지 함께 빌드합니다. `--symlink-install`은 선택 사항입니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/turtlebot3_ws
colcon build --packages-up-to my_yolo_cpp_pkg navigation --symlink-install
source install/setup.bash
```

빌드 후 각 실행 터미널에 아래 환경을 적용합니다. `.bashrc`에 이미 설정되어 있다면
중복 실행하지 않아도 됩니다. PC와 로봇은 같은 네트워크·`ROS_DOMAIN_ID`를 사용합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=0
```

### 로봇 측 실행 — 터미널 1~3

각 터미널에서 로봇에 접속한 뒤 해당 명령을 실행합니다.
접속 예시는 `ssh <사용자명>@<로봇_IP>`이며, 사용자명·주소·카메라 장치는 환경에 맞게 변경합니다.

```bash
# 터미널 1: 로봇 브링업
ros2 launch turtlebot3_bringup robot.launch.py

# 터미널 2: Pi 카메라
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:="/dev/video0" -p image_size:="[640,480]" -p buffer_queue_size:=1

# 터미널 3: 서보·팬틸트 서버
ros2 launch control_motor hardware_bringup.launch.py
```

### PC 측 실행 — 터미널 4~6

각 명령은 별도 터미널에서 실행합니다. TEB 워크스페이스 위치는 설치 환경에 맞춥니다.
Nav2 실행 후 RViz의 **2D Pose Estimate**로 초기 위치를 맞추고, 카메라·YOLO 영상을 확인한 뒤 순찰을 시작합니다.

```bash
# 터미널 4: Nav2 + RViz
source ~/teb_ws/install/setup.bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  map:="$HOME/turtlebot3_ws/recycle_project/map/map2.yaml" \
  params_file:="$HOME/turtlebot3_ws/recycle_project/src/navigation/navigation/waffle_pi.yaml" \
  use_sim_time:=false

# 터미널 5: YOLO 탐지 + 영상 창
ros2 launch my_yolo_cpp_pkg yolo_bringup.launch.py

# 터미널 6: 순찰·수거 시작
ros2 launch navigation navigation.launch.py
```

카메라는 `/image_raw/compressed`를 발행해야 합니다. 별도 영상 뷰어 실행은 필요 없습니다.
실주행에는 `navigation.launch.py`를 사용하며, `main.launch.py`는 Gazebo를 포함하는 구성입니다.
`params_file`을 생략하면 TurtleBot3 설치 경로의 설정을 읽으므로 프로젝트 YAML 변경이 적용되지 않습니다.

## 주요 설정 및 화면 표시

| 설정 | 위치 |
| --- | --- |
| 순찰 좌표·HOME 좌표 | [coverage_node.py](src/navigation/navigation/coverage_node.py)의 `patrol_points`, `home` |
| 종류별 분리수거장 좌표 | [recycle.py](src/navigation/navigation/recycle.py)의 `recycle_points` — can, paper, plastic, trash 순서 |
| Nav2·TEB 주행 설정 | [waffle_pi.yaml](src/navigation/navigation/waffle_pi.yaml) |
| 분류·복귀 구간 설정 | [navigation.launch.py](src/navigation/launch/navigation.launch.py) |

지도와 좌표는 시험 환경 기준이므로 다른 장소에서는 함께 수정해야 합니다.
기본 경로는 `0 → 1 → 2 → 3 → 4 → 5 → HOME`이며, **4번 도착 직후부터 추론을 중지**합니다.
`home_return_start_index=5`는 수신 경로에서 0부터 세는 순번이므로 선택 경로의 순서를 바꾸면 함께 확인합니다.
첫 물체의 접근 분류 기본값은 클래스별 누적 10회, 평균 신뢰도 차이 0.05입니다.

화면의 `DETECTED`는 현재 검출, `SELECTED`는 채택한 수거 종류입니다.
박스·라벨은 선택 대상 하나에만 표시하며, 추론 중지 중에도 `PAUSED` 카메라 화면은 유지됩니다.
