# rt 

This repository contains a CUDA GPU runtime and debugger, using an MLIR frontend.

# Usage

See USAGE.md.

# Building

TableGen, in `mlir/include/Runtime`:

```
/path/to/mlir-tblgen/bin -gen-op-defs RuntimeDialect.td -o RuntimeDialect.cpp.inc -I /path/to/include/dir
/path/to/mlir-tblgen/bin -gen-op-decls RuntimeDialect.td -o RuntimeDialect.h.inc -I /path/to/include/dir
```

To build the project:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR="/path/to/MLIR/install/with/cmake" 
cmake --build build --config Debug
```

Note that the build is configured to "Debug". This is because MLIR compiles to "Debug" by default.
# Running

To run, use the `rt-run` tool with the chosen `.mlir` file.

To see a graph visualization of written `.mlir`, use the `rt-graph` tool.
