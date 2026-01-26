#include "Runtime/RuntimeDialect.h"
#include "mlir/Pass/Pass.h"
#include "adapters.h"
#include "core/allocator.h"

namespace mlir::rt {
    void lowerOperation(mlir::Operation* op, llvm::DenseMap<Value, Tensor*>& tensors) {
        if (auto alloc = dyn_cast<AllocOp>(op)) {
            auto type = alloc.getType();
            auto shape = type.getShape();

            Tensor* t = new Tensor(Allocator::allocate(shape.vec(), sizeof(type)));
            tensors[alloc.getResult()] = t;
            return;
        }

        if (auto mm = dyn_cast<MatMulOp>(op)) {
            Tensor* A = tensors[mm.getOperand(0)];
            Tensor* B = tensors[mm.getOperand(1)];

            auto type = mm.getResult().getType();
            auto shape = type.getShape();

            Tensor* C = new Tensor(Allocator::allocate(shape.vec(), sizeof(type)));
            rt_matrixMul(A, B, C);

            tensors[mm.getResult()] = C;
            return;
        }

        if (auto noise = dyn_cast<NoiseOp>(op)) {
            Tensor* A = tensors[noise.getOperand()];

            rt_generateNoise(A, -10.0f, 10.0f, 42u);
            tensors[noise.getResult()] = A;
            return;
        }

        if (auto add = dyn_cast<AddOp>(op)) {
            Tensor* A = tensors[add.getOperand(0)];
            Tensor* B = tensors[add.getOperand(1)];

            auto type = add.getResult().getType();
            auto shape = type.getShape();

            Tensor* C = new Tensor(Allocator::allocate(shape.vec(), sizeof(type)));
            rt_vectorAdd(A, B, C);

            tensors[add.getResult()] = C;
            return;
        }

        if (auto relu = dyn_cast<ReLUOp>(op)) {
            Tensor* A = tensors[relu.getOperand()];

            rt_vectorReLU(A);
            tensors[relu.getResult()] = A;
            return;
        }
    }

    // TODO: Complete this function, to add an operation to a graph.
    // ----
    //       This will add the given operation, as well as catch its
    //       dependencies and set them as ingoing edges, and set results
    //       as outgoing edges. It will automatically figure out where 
    //       ingoing edges come from, and where outgoing edges go to.
    //       If some outgoing edges have nowhere to go by the end of the
    //       input, delete them.
    void addOpToGraph(mlir::Operation* op, llvm::DenseMap<Value, Tensor*>& tensors) {
        if (auto alloc = dyn_cast<AllocOp>(op)) {
            auto type = alloc.getType();
            auto shape = type.getShape();

            Tensor* t = new Tensor(Allocator::allocate(shape.vec(), sizeof(type)));
            tensors[alloc.getResult()] = t;
            return;
        }

        if (auto mm = dyn_cast<MatMulOp>(op)) {
            Tensor* A = tensors[mm.getOperand(0)];
            Tensor* B = tensors[mm.getOperand(1)];

            auto type = mm.getResult().getType();
            auto shape = type.getShape();

            Tensor* C = new Tensor(Allocator::allocate(shape.vec(), sizeof(type)));
            rt_matrixMul(A, B, C);

            tensors[mm.getResult()] = C;
            return;
        }

        if (auto noise = dyn_cast<NoiseOp>(op)) {
            Tensor* A = tensors[noise.getOperand()];

            rt_generateNoise(A, -10.0f, 10.0f, 42u);
            tensors[noise.getResult()] = A;
            return;
        }

        if (auto add = dyn_cast<AddOp>(op)) {
            Tensor* A = tensors[add.getOperand(0)];
            Tensor* B = tensors[add.getOperand(1)];

            auto type = add.getResult().getType();
            auto shape = type.getShape();

            Tensor* C = new Tensor(Allocator::allocate(shape.vec(), sizeof(type)));
            rt_vectorAdd(A, B, C);

            tensors[add.getResult()] = C;
            return;
        }

        if (auto relu = dyn_cast<ReLUOp>(op)) {
            Tensor* A = tensors[relu.getOperand()];

            rt_vectorReLU(A);
            tensors[relu.getResult()] = A;
            return;
        }

    }
} // namespace mlir::rt
