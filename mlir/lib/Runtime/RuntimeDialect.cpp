#include "Runtime/RuntimeDialect.h"

#include "mlir/IR/DialectImplementation.h"

#define GET_OP_CLASSES
#include "Runtime/RuntimeDialect.cpp.inc"

namespace mlir {
namespace rt {

    // Constructor for the RT dialect.
    RT_Dialect::RT_Dialect(mlir::MLIRContext *context) : mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<RT_Dialect>()) {
      addOperations<
#define GET_OP_LIST
#include "Runtime/RuntimeDialect.cpp.inc"
          >();
    }

} // namespace rt
} // namespace mlir
