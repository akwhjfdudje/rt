#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "RuntimeOps.h.inc"

namespace rt {
class RTDialect : public mlir::Dialect {
public:
  explicit RTDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "rt"; }
  //void initialize() override;
};

// Define the operations
class AllocOp : public mlir::Op<AllocOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static mlir::StringRef getOperationName() { return "rt.alloc"; }

  // Parse and print methods
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type type);
  static mlir::ParseResult parse(mlir::OpAsmParser &parser, mlir::OperationState &result);
  // static void print(mlir::OpAsmPrinter &p, AllocOp op);

  // Additional redundant methods:
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
};

class MatMulOp : public mlir::Op<MatMulOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static mlir::StringRef getOperationName() { return "rt.matmul"; }

  // Parse and print methods
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs, mlir::Type type);
  static mlir::ParseResult parse(mlir::OpAsmParser &parser, mlir::OperationState &result);
  // static void print(mlir::OpAsmPrinter &p, MatMulOp op);

  // Additional redundant methods:
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
};

} // namespace rt
