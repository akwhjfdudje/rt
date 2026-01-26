#include "noise_adapter.h"
#include "noise/noise.cuh"
#include "debug/trace.h"
#include "debug/step.h"
#include "debug/snapshot.h"
#include "debug/guard.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

void rt_generateNoise(Tensor* output, float min_val, float max_val, unsigned int seed) {
    // Start kernel trace
    trace_begin("generateNoise");
    
    debug_step("before generateNoise");
    dump_tensor(*output, "generateNoise:output");
    // Get number of elements
    int N = static_cast<int>(output->numel());
    
    // Allocate temporary host buffer
    std::vector<float> host_buffer(N);
    
    // Generate noise (the function uses GPU internally and copies back to host)
    generateNoise(host_buffer.data(), N, min_val, max_val, seed);
    
    // Copy to device memory
    cudaError_t err = cudaMemcpy(
        output->device,
        host_buffer.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in cudaMemcpy: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in generateNoise: " << cudaGetErrorString(err) << std::endl;
    }
    
    debug_step("after generateNoise");
    dump_tensor(*output, "generateNoise:output");
    
    guard_nan_inf(*output, "generateNoise:output");
    
    trace_end("generateNoise");
}
