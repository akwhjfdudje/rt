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
    LLVMSupport
    MLIRSupport
    MLIRParser
    MLIRDialect
    MLIRAsmParser
    MLIRTableGen
    MLIRTranslateLib
)

add_executable(rt-run tools/rt-run.cpp)

target_link_libraries(rt-run RuntimeMLIR)
