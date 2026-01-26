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
    trace_begin("generateNoise");
    
    debug_step("before generateNoise");
    dump_tensor(*output, "generateNoise:output");

    int N = output->shape[0] * output->shape[1]; // assume output is 2D; need to flatten it
    
    generateNoise(
        reinterpret_cast<float*>(output->device), 
        N, 
        min_val, 
        max_val, 
        seed
    );
    
    cudaDeviceSynchronize();
     
    debug_step("after generateNoise");
    dump_tensor(*output, "generateNoise:output");
    
    guard_nan_inf(*output, "generateNoise:output");
    
    trace_end("generateNoise");
}
