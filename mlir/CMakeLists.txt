set(LLVM_TARGET_DEFINITIONS RuntimeOps.td)

mlir_tablegen(RuntimeOps.h.inc -gen-op-decls)
mlir_tablegen(RuntimeOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(RuntimeOpsIncGen)

add_library(RuntimeMLIR STATIC
    lib/Runtime/RuntimeDialect.cpp
    lib/Runtime/Passes.cpp
)

target_include_directories(RuntimeMLIR PUBLIC
    include
    ../adapters/include
)

target_link_libraries(RuntimeMLIR PUBLIC
    gpu_runtime
    MLIRIR
    MLIRPass
    MLIRFuncDialect
)

add_executable(rt-opt tools/rt-opt.cpp)
target_link_libraries(rt-opt RuntimeMLIR)

add_executable(rt-run tools/rt-run.cpp)
target_link_libraries(rt-run RuntimeMLIR)

add_executable(rt-replay tools/rt-replay.cpp)
target_link_libraries(rt-replay RuntimeMLIR)

