#pragma once
#include "mlir/IR/Types.h"

namespace rt {
class TensorType : public mlir::Type::TypeBase<TensorType, mlir::Type, mlir::TypeStorage> {
public:
    using Base::Base;
    static TensorType get(mlir::MLIRContext* ctx);
};
} // namespace rt

