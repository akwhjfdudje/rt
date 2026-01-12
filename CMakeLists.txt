cmake_minimum_required(VERSION 3.20)
project(gpu_runtime LANGUAGES C CXX CUDA)

# Options
option(GPU_RUNTIME_ENABLE_MLIR "Enable MLIR dialect + tools" ON)
option(GPU_RUNTIME_ENABLE_DEBUG "Enable debug & visualization tools" ON)
option(GPU_RUNTIME_BUILD_EXAMPLES "Build examples" ON)

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Core runtime library
add_library(gpu_runtime STATIC
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/core/graph.cpp
    src/core/op.cpp
    src/core/stream.cpp
    src/runtime_api.cpp
    ${ADAPTER_SRC}
)

file(GLOB_RECURSE ADAPTER_SRC
    "kernel_adapters/src/*.cu"
)
file(GLOB_RECURSE ADAPTER_INCLUDE
    "kernel_adapters/include"
)

target_include_directories(gpu_runtime PUBLIC
    include
    ${ADAPTER_INCLUDE}
)

target_link_libraries(gpu_runtime PUBLIC
    CUDA::cudart
)

# Debug / visualization
if (GPU_RUNTIME_ENABLE_DEBUG)
    target_sources(gpu_runtime PRIVATE
        src/debug/trace.cpp
        src/debug/snapshot.cpp
        src/debug/guard.cpp
        src/debug/diff.cpp
        src/debug/step.cpp
        src/debug/visualize.cpp
        src/debug/record.cpp
    )

    target_compile_definitions(gpu_runtime PUBLIC GPU_RUNTIME_DEBUG=1)
endif()

# MLIR integration
if (GPU_RUNTIME_ENABLE_MLIR)
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Found MLIR: ${MLIR_DIR}")

    include_directories(
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
    )
    add_definitions(${MLIR_DEFINITIONS})

    add_subdirectory(mlir)
endif()

