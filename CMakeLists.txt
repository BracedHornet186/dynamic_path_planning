cmake_minimum_required(VERSION 3.5)
project(dynamic_path_planning LANGUAGES CXX CUDA)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

enable_language(CUDA)

include_directories(
  include
  )

add_executable(mc_collision_monitor
    src/mc_collision_monitor_node.cu
    src/mc_collision_kernel.cu
)

set_target_properties(mc_collision_monitor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(mc_collision_monitor
    cuda
)

ament_target_dependencies(mc_collision_monitor
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
)

install(
  TARGETS
  mc_collision_monitor
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
  scripts/path_publisher.py
  scripts/vel_publisher.py
  scripts/pedestrian_predictor.py
  scripts/mc_collision_monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config  
  launch
  models
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(dynamic_path_planning)

ament_package()
