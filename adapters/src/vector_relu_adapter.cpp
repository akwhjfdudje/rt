#include "vector_relu_adapter.h"
#include "activate/activate.cuh"
#include "debug/trace.h"
#include "debug/step.h"
#include "debug/snapshot.h"
#include "debug/guard.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

void rt_vectorReLU(Tensor* output) {
    trace_begin("vectorReLU");
    
    dump_tensor(*output, "vectorReLU:output");

    debug_step("before vectorReLU");

    int N = output->shape[0] * output->shape[1]; // assume output is 2D; need to flatten it
    
    vectorReLU(
        reinterpret_cast<float*>(output->device), 
        reinterpret_cast<float*>(output->device),
        N
    );
    
    cudaDeviceSynchronize();
     
    dump_tensor(*output, "vectorReLU:output");

    debug_step("after vectorReLU");
    
    guard_nan_inf(*output, "vectorReLU:output");
    
    trace_end("vectorReLU");
}
