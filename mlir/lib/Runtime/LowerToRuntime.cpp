#include "Runtime/RuntimeDialect.h"
#include "runtime_api.h"
#include "debug/trace.h"
#include "debug/snapshot.h"

using namespace mlir;

void lowerRtMatmulOp(Tensor* A, Tensor* B, Tensor* C) {
    rt_matrixMul(A, B, C);
}

