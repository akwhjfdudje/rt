#pragma once
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "core/tensor.h"

#define GET_OP_CLASSES
#include "./RuntimeDialect.h.inc"

namespace rt {
class RT_Dialect : public mlir::Dialect {
public:
  explicit RT_Dialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "rt"; }
  void initialize();
};
void lowerOperation(mlir::Operation* op, llvm::DenseMap<mlir::Value, Tensor*>& tensors);
}
