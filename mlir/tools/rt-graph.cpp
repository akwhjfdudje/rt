#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/ViewOpGraph.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "Runtime/RuntimeDialect.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: rt-graph <file.mlir>\n");
        return 1;
    }

    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<mlir::rt::RT_Dialect>();

    auto module = mlir::parseSourceFile(argv[1], &ctx);
    if (!module) return 1;

    module.get()->getParentRegion()->viewGraph("module");

    return 0;
}

