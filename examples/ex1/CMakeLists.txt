cmake_minimum_required(VERSION 2.8)

project(petscpp-ex1 C CXX)

# Get Eigen, Petsc, and Petsc bin directories from env variables
set(EIGEN_DIR $ENV{EIGEN_DIR})
set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})

# Check if env variables are missing
if (NOT DEFINED EIGEN_DIR)
  message( SEND_ERROR "path to Eigen 'EIGEN_DIR' is not set." )
endif()

include_directories(${EIGEN_DIR})

if (NOT DEFINED PETSC_DIR)
  message( SEND_ERROR "path to petsc 'PETSC_DIR' is not set." )
endif()

if (NOT DEFINED PETSC_ARCH)
  message( SEND_ERROR "path to petsc build 'PETSC_ARCH' is not set." )
endif()

# Load petsc definitions and includes
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules/petsc")
find_package(PETSc REQUIRED)
include_directories(${PETSC_INCLUDES})
add_definitions(${PETSC_DEFINITIONS})

# Build petscpp
if (NOT TARGET petscpp)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Petscpp petscpp)
endif()

# Petscpp include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)

# Source files
set(ex1_source_files
  main.cpp)
add_executable(ex1 ${ex1_source_files})

# Link against petsc and petscpp
target_link_libraries(ex1
  petscpp
  ${PETSC_LIBRARIES}
  ${PETSC_DIR}/${PETSC_ARCH}/lib/libmpich.so
  )
