

cmake_minimum_required(VERSION 3.21)
project(Real_time_Object_2_D_Recognition)

find_package(OpenCV)
include_directories(${/usr/local/include/})

set(CMAKE_CXX_STANDARD 14)

add_executable(Real_time_Object_2_D_Recognition main.cpp test.cpp segmentation.cpp segmentation.h test2.cpp features.cpp features.h csv_util.cpp csv_util.h classify.cpp classify.h)
target_link_libraries(Real_time_Object_2_D_Recognition ${OpenCV_LIBS})