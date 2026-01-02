cmake_minimum_required(VERSION 3.5)
project(dynamic_path_planning LANGUAGES CXX CUDA)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
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

# Find MPPI-Generic
find_package(MPPI-Generic REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

enable_language(CUDA)

include_directories(
  include
  )

add_executable(mppi_controller
    src/mppi_controller_node.cu
)
target_include_directories(mppi_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MPPI_INCLUDE_DIRS}
)
set_target_properties(mppi_controller PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(mppi_controller
  MPPI::MPPI
  Eigen3::Eigen
  ${CUDA_LIBRARIES}
)
ament_target_dependencies(mppi_controller
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
)

install(
  TARGETS
  mppi_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
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
