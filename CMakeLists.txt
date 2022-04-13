# ------------------------------------------------------------------------------
# CMake file for amm
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.8)

message(STATUS "> configuring AMM in (" ${CMAKE_CURRENT_SOURCE_DIR} ")")

project(AMM LANGUAGES CXX VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)

set(PROJ_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE 1)

# ------------------------------------------------------------------------------
# build type

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})
message(STATUS "Compiler:  " ${CMAKE_CXX_COMPILER_ID})

# ------------------------------------------------------------------------------
# compiler configuration
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-O3") #" -w -fpermissive")
endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-fmax-errors=1 -O3") # -w -fpermissive")
endif()

# for development, let's catch all warnings
# add_compile_options(-Wall -Wextra -pedantic -Werror)

# ------------------------------------------------------------------------------
# add VTK

FIND_PACKAGE(VTK REQUIRED)
if(VTK_FOUND)
  message(STATUS "found VTK. Version:" ${VTK_VERSION} ". VTK_DIR: " ${VTK_DIR})

  include(${VTK_USE_FILE})
  add_definitions(-DAMM_USE_VTK=TRUE)
endif(VTK_FOUND)

# ------------------------------------------------------------------------------
# The main app that use amm

add_subdirectory(app)

# ------------------------------------------------------------------------------
