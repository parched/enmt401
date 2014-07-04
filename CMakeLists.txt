cmake_minimum_required(VERSION 3.0)
project(OpticalNavAid)
find_package(OpenCV REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_executable(matcher-simple src/matcher_simple.cpp)
target_link_libraries(matcher-simple ${OpenCV_LIBS})
