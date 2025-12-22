cmake_minimum_required(VERSION 3.5)
project(dynamic_path_planning)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

include_directories(
  include
  )

install(
  PROGRAMS
  scripts/path_publisher.py
  scripts/vel_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config  
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(dynamic_path_planning)

ament_package()
