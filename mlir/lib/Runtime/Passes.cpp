#include "Runtime/RuntimeDialect.h"
#include "adapters/include/matrix_mul_adapter.h"
#include "core/tensor.h"

using namespace mlir;

namespace rt {

void lowerOperation(Operation* op,
                    std::map<Value, Tensor*>& tensors) {
    if (auto alloc = dyn_cast<AllocOp>(op)) {
        auto type = alloc.getType().cast<TensorType>();
        auto shape = type.getShape();

        Tensor* t = new Tensor(shape, sizeof(float));
        tensors[alloc.getResult()] = t;
        return;
    }

    if (auto mm = dyn_cast<MatMulOp>(op)) {
        Tensor* A = tensors[mm.getOperand(0)];
        Tensor* B = tensors[mm.getOperand(1)];

        auto type = mm.getResult().getType().cast<TensorType>();
        auto shape = type.getShape();

        Tensor* C = new Tensor(shape, sizeof(float));
        tensors[mm.getResult()] = C;

        rt_matrixMul(A, B, C);
        return;
    }
}
};


} // namespace rt

