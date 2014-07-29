cmake_minimum_required(VERSION 3.0)
project(OpticalNavAid)
find_package(OpenCV REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Werror -Wpedantic")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_executable(matcher-simple src/matcher_simple.cpp)
target_link_libraries(matcher-simple ${OpenCV_LIBS})

add_executable(optical-nav-aid src/optical_nav_aid.cpp src/matches.cpp src/pose.cpp)
target_link_libraries(optical-nav-aid ${OpenCV_LIBS})
