#include "Runtime/RuntimeDialect.h"
#include "Runtime/RuntimeOps.h"
#include "mlir/IR/Builders.h"

using namespace rt;
using namespace mlir;

RuntimeDialect::RuntimeDialect(MLIRContext* ctx)
    : Dialect(getDialectNamespace(), ctx) {
    addOperations<
#define GET_OP_LIST
#include "RuntimeOps.cpp.inc"
    >();
}

