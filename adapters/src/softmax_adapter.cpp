#include "softmax_adapter.h"
#include "activate/activate.cuh"
#include "debug/trace.h"
#include "debug/step.h"
#include "debug/snapshot.h"
#include "debug/guard.h"
#include <cuda_runtime.h>
#include <iostream>

void rt_softmax(Tensor* output) {
    trace_begin("softmax");
    
    dump_tensor(*output, "softmax:output");

    debug_step("before softmax");

    int batch_size = output->shape[0];
    int features = output->shape[1]; 
    
    softmax(
        reinterpret_cast<float*>(output->device), 
        reinterpret_cast<float*>(output->device),
        batch_size,
        features
    );
    
    cudaDeviceSynchronize();
     
    dump_tensor(*output, "softmax:output");

    debug_step("after softmax");
    
    guard_nan_inf(*output, "softmax:output");
    
    trace_end("softmax");
}
