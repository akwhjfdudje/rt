#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "Runtime/RuntimeDialect.h"
#include "Runtime/LowerToRuntime.h"
#include "core/tensor.h"

#include <map>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: rt-run <file.mlir>\n");
        return 1;
    }

    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<rt::RuntimeDialect>();

    auto module = mlir::parseSourceFile(argv[1], &ctx);
    if (!module) return 1;

    // Maps MLIR SSA values → runtime tensors
    std::map<mlir::Value, Tensor*> tensorMap;

    // Walk ops in program order
    module->walk([&](mlir::Operation* op) {
        rt::lowerOperation(op, tensorMap);
    });

    return 0;
}

