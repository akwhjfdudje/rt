#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "./RuntimeDialect.h.inc"
