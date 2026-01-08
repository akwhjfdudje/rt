#include "Runtime/RuntimeDialect.h"
#include "Runtime/RuntimeOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace rt;

void RuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RuntimeOps.cpp.inc"
      >();
}
