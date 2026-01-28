#include "matrix_mul_adapter.h"
#include "linalg/linalg.cuh"
#include "debug/trace.h"
#include "debug/step.h"
#include "debug/snapshot.h"
#include "debug/guard.h"
#include <cuda_runtime.h>
#include <iostream>

void rt_matrixMul(Tensor* A, Tensor* B, Tensor* C) {
    trace_begin("matrixMul");

    dump_tensor(*A, "matrixMul:A");
    dump_tensor(*B, "matrixMul:B");
    
    debug_step("before matrixMul");
    
    int N = A->shape[0]; // assume all tensors are square NxN

    matrixMul(
        reinterpret_cast<const float*>(A->device),
        reinterpret_cast<const float*>(B->device),
        reinterpret_cast<float*>(C->device),
        N
    );

    cudaDeviceSynchronize();

    dump_tensor(*C, "matrixMul:C");

    debug_step("after matrixMul");

    guard_nan_inf(*C, "matrixMul:C");

    trace_end("matrixMul");
}

