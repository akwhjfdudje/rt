#include "Runtime/RuntimeDialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace rt;

#include "Runtime/RuntimeDialect.cpp.inc"

// Constructor for the RT dialect.
RT_Dialect::RT_Dialect(mlir::MLIRContext *context) : mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<RT_Dialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Runtime/RuntimeDialect.cpp.inc"
      >();
}
