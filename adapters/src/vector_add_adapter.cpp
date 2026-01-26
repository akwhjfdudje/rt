#include "vector_add_adapter.h"
#include "arith/arith.cuh"
#include "debug/trace.h"
#include "debug/step.h"
#include "debug/snapshot.h"
#include "debug/guard.h"
#include <cuda_runtime.h>
#include <iostream>

void rt_vectorAdd(Tensor* A, Tensor* B, Tensor* C) {
    trace_begin("vectorAdd");

    debug_step("before vectorAdd");
    dump_tensor(*A, "vectorAdd:A");
    dump_tensor(*B, "vectorAdd:B"); 
    
    int N = A->shape[0] * A->shape[1]; // get all the bytes

    vectorAdd(
        reinterpret_cast<const float*>(A->device),
        reinterpret_cast<const float*>(B->device),
        reinterpret_cast<float*>(C->device),
        N
    );

    cudaDeviceSynchronize();

    debug_step("after vectorAdd");
    dump_tensor(*C, "vectorAdd:C");

    guard_nan_inf(*C, "vectorAdd:C");

    trace_end("vectorAdd");
}

