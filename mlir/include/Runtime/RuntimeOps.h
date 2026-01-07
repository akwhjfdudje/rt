#pragma once
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

namespace rt {
#define GET_OP_CLASSES
#include "RuntimeOps.h.inc"
}

