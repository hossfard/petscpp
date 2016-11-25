cmake_minimum_required(VERSION 2.8)

project(petscpp)

# Get Eigen dir from env variables
set(EIGEN_DIR $ENV{EIGEN_DIR})

# Check if env variables are missing
if (NOT DEFINED EIGEN_DIR)
  message( SEND_ERROR "path to Eigen 'EIGEN_DIR' is not set." )
endif()

include_directories(${EIGEN_DIR})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/petsc")
find_package(PETSc REQUIRED)
include_directories(${PETSC_INCLUDES})
add_definitions(${PETSC_DEFINITIONS})

set(petscpp_source_files
  BitArray.h
  Petscpp.h
  IndexSet.cpp
  IndexSet.h
  Ksp.cpp
  Ksp.h
  Map.cpp
  Map.h
  Matrix.cpp
  Matrix.h
  MatrixPartitioning.cpp
  MatrixPartitioning.h
  MatrixProxy.cpp
  MatrixProxy.h
  Utility.h
  Vector.cpp
  Vector.h
  )

add_library(petscpp ${petscpp_source_files})

# Make sure the compiler can find include files for our 'petscpp'
# library when other libraries or executables link to it
target_include_directories(petscpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
