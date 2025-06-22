cmake_minimum_required(VERSION 3.15...3.29)
project(
  ${SKBUILD_PROJECT_NAME}
  DESCRIPTION "Compute triangulations of point configurations"
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX)

# Required for TOPCOM
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TOPCOM
add_subdirectory(extern/topcom)
target_compile_definitions(topcom PUBLIC USE_PHIT_PREPROCESSING=false)

# Boost (headers only)
set(Boost_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/extern/boost)

# CGAL
set(CGAL_DISABLE_GMP ON)
set(CGAL_DIR ${CMAKE_SOURCE_DIR}/extern/cgal)
find_package(CGAL REQUIRED CONFIG)

# Eigen
add_library(Eigen3::Eigen INTERFACE IMPORTED)
# Don't use add_subdirectory because it configures stuff we don't want
set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                               ${CMAKE_SOURCE_DIR}/extern/eigen)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(TRIANGULUMANCER_SOURCES
    src/triangulumancer/triangulumancer.cpp
    src/triangulumancer/PointConfiguration.cpp
    src/triangulumancer/Triangulation.cpp
    src/triangulumancer/CGAL.cpp
    src/triangulumancer/TOPCOM.cpp)
pybind11_add_module(triangulumancer ${TRIANGULUMANCER_SOURCES})
target_link_libraries(triangulumancer PRIVATE topcom checkreg CGAL::CGAL
                                              Eigen3::Eigen)
target_include_directories(triangulumancer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(triangulumancer PRIVATE cxx_std_17)
target_compile_definitions(triangulumancer PRIVATE CGAL_EIGEN3_ENABLED)

install(TARGETS triangulumancer DESTINATION .)
