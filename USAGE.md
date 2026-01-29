# MLIR

This project was created primarily to learn how the MLIR framework works, by making a simple dialect along with an interpreter pass that allows users to organize and manage kernel orchestration.

## Dialect

The main part of this runtime is the dialect, which contain ops that map to kernels and memory handling operations.

To add an operation to the dialect, see `mlir/include/Runtime/RuntimeDialect.td` for some example definitions.

## Passes

### Interpreter

The second part of the runtime is an interpreter pass that runs every op in a provided `.mlir` file in sequence, providing debug information along the way.

To add to the interpreter pass, see `mlir/lib/Runtime/Passes.cpp` for examples on how to add operations to the pass.

### Fusion (WIP)

# Adapters

To add a kernel adapter, see the `adapters/` directory for examples on provided adapters for select kernels.

