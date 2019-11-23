cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_fizz)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_PREFIX_PATH ./libtorch)
find_package(Torch REQUIRED)

add_executable(torch_fizz torch_fizz.cc)
target_link_libraries(torch_fizz "${TORCH_LIBRARIES}")
set_property(TARGET torch_fizz PROPERTY CXX_STANDARD 17)
set_property(TARGET torch_fizz PROPERTY CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -g -fno-builtin")
