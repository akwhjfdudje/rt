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
    # REQUIRED core
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/core/graph.cpp
    src/core/op.cpp
    src/core/stream.cpp
    src/runtime_api.cpp
)

target_include_directories(gpu_runtime PUBLIC
    include
    kernels/include
)

target_link_libraries(gpu_runtime PUBLIC
    CUDA::cudart
)

# CUDA kernels
add_library(cuda_kernels STATIC
    kernels/src/matmul.cu
    kernels/src/add.cu
    kernels/src/relu.cu
    kernels/src/noise.cu
    kernels/src/heightmap.cu
    kernels/src/fuse.cu
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cuda_kernels PUBLIC
    kernels/include
)

target_link_libraries(cuda_kernels PUBLIC
    CUDA::cudart
)

target_link_libraries(gpu_runtime PUBLIC cuda_kernels)

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

# CPU reference kernels
add_library(cpu_reference STATIC
    src/cpu/matmul_cpu.cpp
    src/cpu/noise_cpu.cpp
    src/cpu/fuse_cpu.cpp
)

target_include_directories(cpu_reference PUBLIC include)
target_link_libraries(gpu_runtime PUBLIC cpu_reference)

# MLIR integration
if (GPU_RUNTIME_ENABLE_MLIR)
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Found MLIR: ${MLIR_DIR}")

    include_directories(${MLIR_INCLUDE_DIRS})
    add_definitions(${MLIR_DEFINITIONS})

    add_subdirectory(mlir)
endif()

# Examples
if (GPU_RUNTIME_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

