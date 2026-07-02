cmake_minimum_required(VERSION 4.0)
project(the-cat LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE release)

add_library(the-catalogue SHARED the_catalogue.cpp read_cpu.cpp)
target_compile_definitions(the-catalogue PUBLIC STANDALONE)
set_target_properties(the-catalogue PROPERTIES SUFFIX ".so" PREFIX "" CXX_STANDARD 20)
target_link_libraries(the-catalogue PRIVATE arbor::arbor)
