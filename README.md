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

## YOLO 실행과 화면

```bash
ros2 launch my_yolo_cpp_pkg yolo_bringup.launch.py
```

검출 노드와 로컬 영상 창이 함께 실행됩니다. 분류·YOLO 추론은
`classified_object_info_node`에서만 수행하고, `best_yolo_node`는
`/yolo/image/compressed`의 결과 영상을 표시합니다. 웹 화면도 같은 영상을 사용합니다.

- 초록색 `TARGET`: 제어용 검출 정보로 전달한 대상
- 노란색 `PENDING`: 최초 발견 후 연속 감지를 확인 중인 후보

박스와 클래스·신뢰도 라벨은 선택된 대상 하나에만 표시합니다. 다른 검출 물체의
박스·글자는 그리지 않으며, 추론 결과와 제어용 대상 선택은 유지합니다.

화면 상단의 고정된 두 번째 줄에 `DETECTED: PAPER 0.87 | SELECTED: TRASH`처럼
현재 제어용으로 선택한 검출의 클래스·신뢰도와 navigation이 채택한 수거 종류를 표시합니다.
글자 색은 노란색으로 고정하며, 두 클래스가 다르면 첫 줄에 `CLASS DIFF`를 표시합니다.
`SELECTED`는 실제 하역 종류에 사용하는
저장 값이며, 추적 중 예측이 달라져도 바뀌지 않습니다. 채택 전에는 `-`로 표시하고,
첫 수거 실패·취소 또는 하역 완료 시 해제합니다. 이미 적재한 물품이 있으면 그 종류를 유지합니다.
단, 첫 번째 물체의 수거 성공 시 접근 중 평균 신뢰도로 분류하거나, 애매하면
수거함에서 재확인하여 `SELECTED`와 실제 하역 종류를 함께 보정합니다.
추론 중지 중에도 채택 종류는 표시하며 현재 검출은 `-`로 표시합니다.
이 상태는 `/selected_recycle_class` (`std_msgs/msg/Int32`, `-1`은 없음)의 마지막 값을
새 구독자도 받도록 발행합니다. 글자 크기는 기존과 같고, 상단·박스 라벨의 사각형 배경은
사용하지 않습니다. 노란 글자에 얇은 검정 외곽선과 가장자리 보정을 적용하며 JPEG 품질은 90입니다.
영상처럼 배경이 변하는 경우의 가독성을 위해
[W3C의 글자 외곽선 방법](https://www.w3.org/WAI/WCAG21/Techniques/general/G18)을 참고했습니다.

영상 표시·압축은 최대 5Hz로 제한하며, 제어용 검출은 입력 프레임마다 처리합니다.
영상 수신 전에는 대기 창이 나타납니다. 창 닫기, `q`, `Esc`는 로컬 뷰어만 종료하며
검출과 웹 영상 발행은 계속됩니다.

### 첫 물체의 접근 분류와 수거함 재확인

최초 채택한 종류는 접근 중 임시로 유지합니다. 대상 채택 후 새로 처리한 관측을
수거 성공 응답까지 누적하며, 클래스별 검출 수가 10회 이상인 후보끼리 평균 신뢰도를
비교합니다. 1·2위 평균 차이가 0.05 이상이면 1위로 확정하고, 조건을 충족한 클래스가
하나뿐이면 그 클래스로 확정합니다. 10회는 연속 횟수가 아닌 누적 횟수이고,
횟수를 채우자마자 조기에 확정하거나 추가 대기하지 않습니다.

검출 수가 부족하거나 평균 차이가 작으면 기존 3초 수거함 확인 구간에서 재분류합니다.
틸트 성공 응답 후 0.4초가 지난 다음 PC에서 처리하기 시작한 영상 중, 같은 위치의 유효 검출이
최소 3회이고 그중 한 종류가 80% 이상이면 채택 종류를 확정·보정합니다.
접근 중 이미 확정한 클래스는 수거함 관측으로 덮어쓰지 않으며 적재량 확인 동작은 유지합니다.
이후 같은 종류를 추가 수거하는 필터와 하역 목적지, 화면 상단 `SELECTED`가 함께 바뀝니다.
검출 부족·동률·분류 혼동·틸트 실패 시에는 기존 종류를 유지하고 이유를 로그에 남깁니다.
두 번째 물체부터는 이미 담긴 물체와 혼동할 수 있어 재분류하지 않습니다.
추가 추론이나 수거 전 대기는 없으며 기존 적재량 판정은 유지합니다.

`auto_nav`의 시작 시 ROS 파라미터는 다음과 같습니다.

| 파라미터 | 기본값 | 의미 |
| --- | --- | --- |
| `approach_min_samples` | `10` | 접근 평균 비교에 참여할 클래스별 최소 누적 검출 수 |
| `approach_mean_margin` | `0.05` | 유효 후보 1·2위의 평균 신뢰도 차이 하한 |
| `basket_min_samples` | `3` | 필요한 유효 검출 수, 최소 3 |
| `basket_agreement_ratio` | `0.8` | 가장 많이 검출된 클래스의 비율, 0.5 초과~1.0 |
| `basket_settle_sec` | `0.4` | 틸트 성공 응답 후 안정화 시간, 0 이상~3초 미만 |

접근 분류 설정은 `ros2 launch navigation navigation.launch.py approach_min_samples:=10
approach_mean_margin:=0.05`로 지정할 수 있습니다. 기본값은 시험용이며 실제 주행 기록으로
조정해야 합니다. `접근 분류 확정/보류` 로그에는 클래스별 횟수·비율·평균과 판단 이유가 남습니다.
비율은 선택 대상의 유효 검출 기준이며, 신뢰도는 정답 확률이 아닙니다. 같은 수거 시도의
선택 대상을 누적하는 방식으로, 여러 물체 사이에서 대상이 바뀌는 상황까지 식별하지는 않습니다.

`/yolo/class_observation` (`vision_msgs/msg/Detection2DArray`)는 기존 YOLO 결과에서 연속 검출
필터 전의 선택 대상 하나를 전달합니다. 바깥 `header`는 추론 시작 전 PC 수신 시각이며,
개별 검출에는 점수·좌표·원본 카메라 촬영 시각을 보존합니다. 수거함 투표는 기존 수거 종류
필터와 독립적이며, 첫 관측 중심의 50px 이내(640×480 기준)인 재활용 클래스만 셉니다.
틸트 안정화 여부는 같은 PC에서 실행하는 YOLO와 AutoNav의 시각으로 비교하므로
Pi 카메라와 PC의 시각 차이 때문에 모든 관측이 제외되지는 않습니다.
추론 완료 시각으로 과거 프레임을 새 프레임처럼 처리하지 않으며, 카메라 촬영 시각은
중복·역순 확인에만 사용합니다. 촬영 시각이 0이면 PC 수신 시각으로 중복 관측을 제외합니다.
`수거함 분류 확정: trash → paper (4/5회 일치)` 또는 `수거함 분류 유지: ...` 로그로
보정 여부를 확인할 수 있습니다. 유지 로그에는 틸트 응답 대기·시간 조건·중복·다른 위치 등
관측 제외 이유도 남습니다. 토픽 형식이 바뀌었으므로 두 패키지를 함께 빌드하고 재시작해야 합니다.

### 분리수거장 이동 중 추론 중지

`navigation`이 분리수거장 이동을 요청하면 사전 분류와 YOLO 모델 실행을 모두 중지합니다.
하역과 수거 Action 내부의 HOME 복귀까지 중지를 유지하고, 추론 ON 응답을 확인한 뒤
순찰을 재개합니다. 최초 순찰도 추론 제어 서비스가 준비된 뒤 시작합니다.
순찰·물체 추적·수거 직후 3초간 적재 상태 확인에는 기존처럼 추론이 필요합니다.

중지 중에도 기존 창과 웹에는 `PAUSED | Inference off`가 표시된 카메라 영상을
기본 5Hz로 전달합니다. 모델은 메모리에 유지하며 카메라 노드도 계속 실행합니다.
따라서 주로 PC의 추론 연산 부하가 줄고, Pi의 카메라 전송은 계속됩니다.
별도 화면 실행이나 추가 명령은 필요하지 않습니다.

`set_inference_enabled` (`std_srvs/srv/SetBool`) 서비스는 추적 모드와 별개입니다.
AutoNav가 원하는 상태를 1초마다 갱신하며, YOLO 재시작 시에도 다시 전달합니다.
OFF 갱신이 5초간 없으면 YOLO가 자동으로 ON으로 복구하므로 네비게이션 종료 후에도
추론이 영구 중지되지 않습니다. 실행 중인 추적은 OFF 요청보다 우선합니다.
하역 요청 거절·실패·취소 시에도 ON을 요청합니다. 통신 실패는 경고와 함께 재시도하며,
순찰 재개용 ON 응답을 받지 못한 동안에는 순찰 이동을 새로 시작하지 않습니다.

검출 노드의 ROS 파라미터 `paused_image_hz`(기본 `5.0`)와
`inference_pause_timeout_sec`(기본 `5.0`)로 화면 빈도와 중지 갱신 만료 시간을
조정할 수 있습니다. 기본값으로 사용하면 됩니다. 이 만료 시간은 **실패 후 재탐지
제한 시간이 아니며**, 정상적인 추론 ON 요청은 즉시 중지를 해제합니다.
전환 전에 쌓인 카메라 콜백과 최초 발견 확인 횟수는 재사용하지 않습니다.

변경 적용 시 PC에서 두 패키지를 함께 빌드하고 YOLO와 navigation을 다시 실행합니다.

```bash
cd ~/turtlebot3_ws
colcon build --packages-select my_yolo_cpp_pkg navigation --symlink-install
source install/setup.bash
```

### TEB 주행 설정

`src/navigation/navigation/waffle_pi.yaml`은 TEB 설정입니다. 후진 속도 상한
`max_vel_x_backwards: 0.01`보다 작은 `penalty_epsilon: 0.005`를 명시하여
최적화 여유값과 속도 제한의 충돌을 해소합니다.

Humble에서 기존 `turtlebot3_navigation2 navigation2.launch.py` 명령은 기본적으로
`turtlebot3_navigation2` 설치 경로의 `share/turtlebot3_navigation2/param/humble/waffle_pi.yaml`을
읽습니다. 프로젝트 파일만 수정해도 이 설치 파일이 자동으로 바뀌지는 않습니다.
현재 주행 PC에는 같은 값을 설치 파일에도 적용했습니다. 설치 파일 직접 수정은 빌드 없이
Nav2 재시작으로 반영되지만, 해당 패키지를 재빌드하면 덮어써질 수 있습니다.
다른 환경에서 저장소의 설정을 사용하려면 다음과 같이 파일을 명시합니다.

```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  map:="$HOME/map2.yaml" \
  params_file:="$HOME/turtlebot3_ws/recycle_project/src/navigation/navigation/waffle_pi.yaml"
```

## 이동 요청 중 물체 검출

순찰 이동 요청의 수락을 기다리는 동안 발견한 물체는 최신 관측으로 보관합니다.
이동이 수락되면 유효한 관측을 사용해 이동 취소를 요청하고, 이동 종료를 확인한 뒤
수거 Action을 시작합니다. STOP·배터리 부족은 보관된 검출보다 우선합니다.
수거 실패 후 재시도를 늦추는 시간 제한은 두지 않습니다.

`pending_detection_max_age_sec`의 기본값은 `2.0`초입니다. 저장한 관측을 사용할 수 있는
최대 나이이며, 이 시간 동안 기다리는 설정은 아닙니다. 미검출이나 잘못된 좌표를 새로
수신하면 보관한 관측을 지웁니다. 별도 설정 없이 기본값으로 실행할 수 있습니다.

필요하면 `navigation.launch.py` 또는 `main.launch.py` 실행 시 조정할 수 있습니다.

```bash
ros2 launch navigation navigation.launch.py pending_detection_max_age_sec:=2.0
```

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

## Requirements 
- Ubuntu 22.04 
- ROS2 Humble 
- TurtleBot3 Navigation2
- Gazebo 
- RViz2 
- Python 3 
- Ultralytics YOLOv8n
- OpenCV
