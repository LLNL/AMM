# ------------------------------------------------------------------------------
# CMake file for amm-cli
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.8)

# ------------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${PROJ_ROOT_DIR} ${PROJ_ROOT_DIR}/amm)

FILE(GLOB HEADERS ${PROJ_ROOT_DIR}/amm/*.hpp
                  ${PROJ_ROOT_DIR}/amm/types/*.hpp
                  ${PROJ_ROOT_DIR}/amm/containers/*.hpp
                  ${PROJ_ROOT_DIR}/amm/precision/*.hpp
                  ${PROJ_ROOT_DIR}/amm/utils/*.hpp
                  ${PROJ_ROOT_DIR}/amm/tree/*.hpp
                  ${PROJ_ROOT_DIR}/amm/creator/*.hpp
                  ${PROJ_ROOT_DIR}/amm/vtk/*.hpp)

FILE(GLOB HEADERS1 ${PROJ_ROOT_DIR}/wavelets/*.hpp)
FILE(GLOB HEADERS2 ${PROJ_ROOT_DIR}/streams/*.hpp)
FILE(GLOB HEADERS3 ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

SET(SOURCES ${PROJ_ROOT_DIR}/wavelets/utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/cli)

# ------------------------------------------------------------------------------
set(TRG_CLI amm-cli)
add_executable(${TRG_CLI} ${HEADERS} ${HEADERS1} ${HEADERS2} ${HEADERS3} ${SOURCES})

target_link_libraries(${TRG_CLI} LINK_PUBLIC ${VTK_LIBRARIES})
install(TARGETS ${TRG_CLI} DESTINATION bin)
# ------------------------------------------------------------------------------
