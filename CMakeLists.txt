cmake_minimum_required(VERSION 3.14...3.22)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")

project(sap
        VERSION 0.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(arbor
             REQUIRED)
           
find_package(pugixml
             REQUIRED)

find_package(nlohmann_json
             REQUIRED)

find_package(Eigen3
             REQUIRED)

find_package(OpenMP)

add_subdirectory(cat)
    
add_library(sap INTERFACE src/init.hpp
                          src/adi.hpp
                          src/common.hpp
                          src/integrate.hpp
                          src/react.hpp
                          src/grid.hpp
                          src/diff.hpp
                          src/shm.hpp
                          src/state.hpp
                          src/ros.hpp)

target_link_libraries(sap INTERFACE OpenMP::OpenMP_CXX)
target_link_libraries(sap INTERFACE Eigen3::Eigen)

# demo 
add_executable(mrf src/mrf.cpp)
target_link_libraries(mrf PRIVATE arbor::arborio arbor::arbor sap)
add_dependencies(mrf the-catalogue)
