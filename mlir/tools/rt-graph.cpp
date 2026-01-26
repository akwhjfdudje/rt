#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/IR/AsmState.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/ViewOpGraph.h"

// If you have custom dialects
#include "Runtime/RuntimeDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  if (argc < 2) {
    llvm::errs() << "usage: mlir-view-op-graph <file.mlir>\n";
    return 1;
  }

  // --- MLIR context ---
  MLIRContext ctx;
  ctx.loadAllAvailableDialects();
  ctx.getOrLoadDialect<mlir::rt::RT_Dialect>();

  // --- Parse file ---
  llvm::SourceMgr sourceMgr;
  auto file = mlir::openInputFile(argv[1]);
  if (!file) {
    llvm::errs() << "failed to open file: " << argv[1] << "\n";
    return 1;
  }

  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    llvm::errs() << "failed to parse MLIR file\n";
    return 1;
  }

  // --- Visualize ---
  // This will generate DOT and invoke `dot` to show the graph.
  // One graph per region.
  ViewOpGraph(*module, /*shortNames=*/false);

  return 0;
}

