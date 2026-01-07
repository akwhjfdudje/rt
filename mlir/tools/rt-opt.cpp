#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "Runtime/Passes.h"
#include <iostream>

int main(int argc, char** argv) {
    mlir::MLIRContext ctx;
    auto module = mlir::parseSourceFile("example.mlir", &ctx);
    if (!module) return 1;

    mlir::PassManager pm(&ctx);
    pm.addPass(rt::createLowerRtPass());

    if (mlir::failed(pm.run(*module))) {
        std::cerr << "Pass failed!\n";
        return 1;
    }

    module->print(llvm::outs());
    return 0;
}

