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
    
    // Get number of elements
    int N = static_cast<int>(output->numel());
    
    if (N > 0) {
        // Allocate temporary host buffer
        std::vector<float> host_buffer(N);
        
        // Generate noise (the function uses GPU internally and copies back to host)
        generateNoise(host_buffer.data(), N, min_val, max_val, seed);
        
        // Copy to device memory
        cudaMemcpy(
            output->device,
            host_buffer.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice
        );
        
        cudaDeviceSynchronize();
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in generateNoise: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    debug_step("after generateNoise");
    dump_tensor(*output, "generateNoise:output");
    
    guard_nan_inf(*output, "generateNoise:output");
    
    trace_end("generateNoise");
}
