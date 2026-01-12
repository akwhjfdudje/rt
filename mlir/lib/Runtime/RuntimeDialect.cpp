#include "Runtime/RuntimeDialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace rt;

#include "./RuntimeDialect.cpp.inc"

void RuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Runtime/RuntimeDialect.cpp.inc"
      >();
}

// Constructor for the RT dialect.
RTDialect::RTDialect(mlir::MLIRContext *context) : mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<RTDialect>()) {
  addOperations<AllocOp, MatMulOp>();
}

// Build an AllocOp operation
void AllocOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type type) {
  state.addTypes(type);
}

mlir::ParseResult AllocOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::Type type;
  if (parser.parseColonType(type))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

void AllocOp::print(mlir::OpAsmPrinter &p, AllocOp op) {
  p << "rt.alloc" << " : " << op.getResultTypes(0);
}

void MatMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs, mlir::Type type) {
  state.addOperands(lhs);
  state.addOperands(rhs);
  state.addTypes(type);  // The result type matches the lhs type
}

mlir::ParseResult MatMulOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
  mlir::Type type;

  if (parser.parseOperand(lhs) || parser.parseComma() || parser.parseOperand(rhs) || parser.parseColonType(type))
    return mlir::failure();

  result.addOperands(lhs);
  result.addOperands(rhs);
  result.addTypes(type);
  return mlir::success();
}

void MatMulOp::print(mlir::OpAsmPrinter &p, MatMulOp op) {
  p << "rt.matmul " << op.getOperand(0) << ", " << op.getOperand(1) << " : " << op.getResultType(0);
}

