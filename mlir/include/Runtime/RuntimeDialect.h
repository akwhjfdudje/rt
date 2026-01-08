#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"

namespace rt {
class RTDialect : public mlir::Dialect {
public:
  explicit RTDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "rt"; }
  void initialize() override;
};

// Define the operations
class AllocOp : public mlir::Op<AllocOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static mlir::StringRef getOperationName() { return "rt.alloc"; }

  // Parse and print methods
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type type);
  static mlir::ParseResult parse(mlir::OpAsmParser &parser, mlir::OperationState &result);
  static void print(mlir::OpAsmPrinter &p, AllocOp op);
};

class MatMulOp : public mlir::Op<MatMulOp, mlir::OpTrait::TwoOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static mlir::StringRef getOperationName() { return "rt.matmul"; }

  // Parse and print methods
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs);
  static mlir::ParseResult parse(mlir::OpAsmParser &parser, mlir::OperationState &result);
  static void print(mlir::OpAsmPrinter &p, MatMulOp op);
};

} // namespace rt
