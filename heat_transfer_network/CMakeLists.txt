cmake_minimum_required(VERSION 3.3.1)

project(Solution_heat_transfer_network LANGUAGES CXX C)

#include(Config)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(FindMKL)
include(FindPETSc)
#find_package(MKL REQUIRED)
find_package(PETSc REQUIRED)

add_executable (heat_transfer_network main.cpp)

target_link_libraries(
    heat_transfer_network
    PUBLIC
        PETSC::PETSC
)