cmake_minimum_required(VERSION 3.3.1)

project(petsc_mpi LANGUAGES CXX C)

#include(Config)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(FindMKL)
include(FindPETSc)
#find_package(MKL REQUIRED)
find_package(PETSc REQUIRED)

add_executable (ex1 ex1.c)

add_executable (ref_case ref_case.c HeatTransfer1Dimpl.h HeatTransfer1Dimpl.c include_patch_pack.h)

add_executable (network network.c)

add_executable (pipes1 pipes1.c pipe.h pipeImpls.c pipeInterface.c wash.h)

target_link_libraries(
    ex1
    PUBLIC
        PETSC::PETSC
)
target_link_libraries(
	ref_case
    PUBLIC
        PETSC::PETSC	
)

target_link_libraries(
	network
    PUBLIC
        PETSC::PETSC	
)

target_link_libraries(
	pipes1
    PUBLIC
        PETSC::PETSC	
)