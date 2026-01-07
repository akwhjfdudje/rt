#include "mlir/Pass/Pass.h"
#include "Runtime/RuntimeOps.h"

namespace rt {
struct LowerRtPass : public mlir::PassWrapper<LowerRtPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override {
        auto module = getOperation();
        // Iterate ops and call lowering functions
        module.walk([&](mlir::Operation *op) {
            if (auto noiseOp = llvm::dyn_cast<NoiseOp>(op)) {
                // Lower to runtime
                // fill in Tensor* and seed from operands
            }
        });
    }
};
} // namespace rt

std::unique_ptr<mlir::Pass> createLowerRtPass() {
    return std::make_unique<rt::LowerRtPass>();
}

