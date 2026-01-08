#include "Runtime/RuntimeDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"

namespace rt {

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
  p << "rt.alloc" << " : " << op.getType();
}

void MatMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
  state.addOperands(lhs);
  state.addOperands(rhs);
  state.addTypes(lhs.getType());  // The result type matches the lhs type
}

mlir::ParseResult MatMulOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType lhs, rhs;
  mlir::Type type;

  if (parser.parseOperand(lhs) || parser.parseComma() || parser.parseOperand(rhs) || parser.parseColonType(type))
    return mlir::failure();

  result.addOperands(lhs);
  result.addOperands(rhs);
  result.addTypes(type);
  return mlir::success();
}

void MatMulOp::print(mlir::OpAsmPrinter &p, MatMulOp op) {
  p << "rt.matmul " << op.getOperand(0) << ", " << op.getOperand(1) << " : " << op.getType();
}

} // namespace rt

