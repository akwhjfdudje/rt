#include "matrix_mul_adapter.h"
#include "linalg/linalg.cuh"
#include "debug/trace.h"
#include "debug/snapshot.h"
#include "debug/guard.h"
#include <cuda_runtime.h>

void rt_matrixMul(Tensor* A, Tensor* B, Tensor* C) {
    // Start kernel trace
    trace_begin("matrixMul");
    dump_tensor(*A, "matrixMul:A");
    dump_tensor(*B, "matrixMul:B");
    
    debug_step("before matrixMul");

    
    // Launch the kernel
    int N = A->shape[0]; // assume square NxN
    matrixMulKernel<<<dim3((N+15)/16, (N+15)/16), dim3(16, 16)>>>(
        reinterpret_cast<const float*>(A->device),
        reinterpret_cast<const float*>(B->device),
        reinterpret_cast<float*>(C->device),
        N
    );

    cudaDeviceSynchronize();

    debug_step("after matrixMul");
    dump_tensor(*C, "matrixMul:C");

    guard_nan_inf(*C, "matrixMul:C");

    trace_end("matrixMul");
}

