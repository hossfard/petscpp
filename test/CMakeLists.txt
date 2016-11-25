cmake_minimum_required(VERSION 2.8)

project(petscpp-tests C CXX)

# Required for GTEST
find_package(Threads REQUIRED)

# Get eigen and google-test directory from env variables
set(GTEST_DIR $ENV{GTEST_DIR})
set(EIGEN_DIR $ENV{EIGEN_DIR})

# Check if env variables are missing
if (NOT DEFINED EIGEN_DIR)
  message( SEND_ERROR "path to Eigen 'EIGEN_DIR' is not set." )
endif()
if (NOT DEFINED GTEST_DIR)
  message( SEND_ERROR "path to google-test 'GTEST_DIR' is not set." )
endif()

# Google-test headers
include_directories(${GTEST_DIR}/include)

# Eigen directory
include_directories(${EIGEN_DIR})

# Build petscpp
if (NOT TARGET petscpp)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Petscpp petscpp)
endif() 

# Petscpp include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

# Add additional path to cmake module
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules/petsc")

# Get PETSC variables and definitions
find_package(PETSc REQUIRED)
include_directories(${PETSC_INCLUDES})
add_definitions(${PETSC_DEFINITIONS})

# Test files
set(petscpp_test_source_files
  main.cpp
  utility_unittest.cpp
  vector_unittest.cpp
  matrix_unittest.cpp
  )

add_executable(petscpp-tests ${petscpp_test_source_files})

target_link_libraries(petscpp-tests petscpp
  ${PETSC_LIBRARIES}
  # ${GTEST_DIR}/lib/libgtest.a
  ${GTEST_DIR}/lib/libgtest_main.a
  ${CMAKE_THREAD_LIBS_INIT}
  ${PETSC_DIR}/${PETSC_ARCH}/lib/libmpich.so
  )

# Copy helper scripts to build directory
configure_file(runall.sh runall.sh COPYONLY)
