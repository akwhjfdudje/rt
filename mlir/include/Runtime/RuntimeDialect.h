#pragma once
#include "mlir/IR/Dialect.h"

namespace rt {
class RuntimeDialect : public mlir::Dialect {
public:
    explicit RuntimeDialect(mlir::MLIRContext *ctx);
    static llvm::StringRef getDialectNamespace() { return "rt"; }
};
} // namespace rt

