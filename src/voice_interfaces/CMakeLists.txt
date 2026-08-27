cmake_minimum_required(VERSION 3.8)
project(voice_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# srv 및 action 파일을 C++ / Python 코드로 자동 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AudioSpeech.srv"
  DEPENDENCIES std_msgs
)

# 다른 패키지가 이 인터페이스를 사용할 수 있도록 런타임 의존성 내보내기
ament_export_dependencies(rosidl_default_runtime)

ament_package()