#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/LogicalResult.h"
#include "core/tensor.h"

#include "Runtime/RuntimeDialect.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: rt-run <file.mlir>\n");
        return 1;
    }

    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<mlir::rt::RT_Dialect>();

    auto module = mlir::parseSourceFile(argv[1], &ctx);
    if (!module) return 1;

    // Maps MLIR SSA values to runtime tensors
    llvm::DenseMap<mlir::Value, Tensor*> tensors;

    // Walk ops in program order
    module->walk([&](mlir::Operation* op) {
            mlir::rt::lowerOperation(op, tensors);
    });

    return 0;
}

