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
        Tensor* C = tensors[mm.getResult(0)];

        rt_matrixMul(A, B, C);
        return;
    }
}

struct RuntimeExecPass
    : public PassWrapper<RuntimeExecPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::map<Value, Tensor*> tensors;

    module.walk([&](Operation *op) {
      rt::lowerOperation(op, tensors);
    });
  }
};


} // namespace rt

