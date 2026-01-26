#include "Runtime/RuntimeDialect.h"
#include "mlir/Pass/Pass.h"
#include "matrix_mul_adapter.h"
#include "noise_adapter.h"
#include "core/allocator.h"

namespace mlir::rt {
    void lowerOperation(mlir::Operation* op, llvm::DenseMap<Value, Tensor*>& tensors) {
        if (auto alloc = dyn_cast<AllocOp>(op)) {
            auto type = alloc.getType();
            auto shape = type.getShape();

            Tensor t = Allocator::allocate(shape.vec(), sizeof(type));
            tensors[alloc.getResult()] = &t;
            return;
        }

        if (auto mm = dyn_cast<MatMulOp>(op)) {
            Tensor* A = tensors[mm.getOperand(0)];
            Tensor* B = tensors[mm.getOperand(1)];

            auto type = mm.getResult().getType();
            auto shape = type.getShape();

            Tensor C = Allocator::allocate(shape.vec(), sizeof(type));
            rt_matrixMul(A, B, &C);

            tensors[mm.getResult()] = &C;
            return;
        }

        if (auto noise = dyn_cast<NoiseOp>(op)) {
            Tensor* A = tensors[noise.getOperand()];

            rt_generateNoise(A);
            tensors[noise.getResult()] = A;
            return;
        }
    }

} // namespace mlir::rt
