cmake_minimum_required(VERSION 3.24...4.1)
project(
  ${SKBUILD_PROJECT_NAME}
  DESCRIPTION "Compute triangulations of point configurations"
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX)

# Required for TOPCOM
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TOPCOM
include(FetchContent)
FetchContent_Declare(
  topcom
  GIT_REPOSITORY https://github.com/ariostas/TOPCOM.git
  GIT_TAG main)
FetchContent_MakeAvailable(topcom)
target_compile_definitions(topcom PUBLIC USE_PHIT_PREPROCESSING=false)

# Search conda paths
if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# Boost (headers only)
find_package(Boost CONFIG REQUIRED)

# CGAL
set(CGAL_DISABLE_GMP ON)
find_package(CGAL CONFIG REQUIRED)

# Eigen
find_package(Eigen3 CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(TRIANGULUMANCER_SOURCES
    src/triangulumancer/triangulumancer.cpp
    src/triangulumancer/PointConfiguration.cpp
    src/triangulumancer/VectorConfiguration.cpp
    src/triangulumancer/Triangulation.cpp
    src/triangulumancer/CGAL.cpp
    src/triangulumancer/TOPCOM.cpp)
pybind11_add_module(triangulumancer ${TRIANGULUMANCER_SOURCES})
target_link_libraries(triangulumancer PRIVATE topcom checkreg CGAL::CGAL
                                              Eigen3::Eigen Boost::headers)
target_include_directories(triangulumancer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(triangulumancer PRIVATE cxx_std_17)
target_compile_definitions(triangulumancer PRIVATE CGAL_EIGEN3_ENABLED)

install(TARGETS triangulumancer DESTINATION .)
