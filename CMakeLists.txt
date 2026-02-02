cmake_minimum_required(VERSION 3.20)
project(runtime LANGUAGES C CXX CUDA)

# Options
option(RUNTIME_ENABLE_MLIR "Enable MLIR dialect + tools" ON)
option(RUNTIME_ENABLE_DEBUG "Enable debug & visualization tools" ON)

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE ADAPTER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/src/*.cpp
)

file(GLOB_RECURSE CORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
)

# Core runtime library
add_library(runtime STATIC
    ${CORE_SRC}
    src/runtime_api.cpp
    ${ADAPTER_SRC}
)

target_include_directories(runtime PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_kernels/include
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/include
)

target_link_libraries(runtime PUBLIC
    CUDA::cudart
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_kernels/build/Release/cuda_kernels.lib
)

# Debug / visualization
if (RUNTIME_ENABLE_DEBUG)
    target_sources(runtime PRIVATE
        src/debug/trace.cpp
        src/debug/snapshot.cpp
        src/debug/guard.cpp
        src/debug/diff.cpp
        src/debug/step.cpp
        src/debug/visualize.cpp
        src/debug/record.cpp
    )

    target_compile_definitions(runtime PUBLIC RUNTIME_DEBUG=1)
endif()

# MLIR integration
if (RUNTIME_ENABLE_MLIR)
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Found MLIR: ${MLIR_DIR}")

    include_directories(
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
    )
    add_definitions(${MLIR_DEFINITIONS})

    add_subdirectory(mlir)
endif()

