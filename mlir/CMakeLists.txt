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
    MLIRAnalysis
    MLIRAsmParser
    MLIRDialect
    MLIRDialectUtils
    MLIRIR
    MLIRObservers
    MLIRParser
    MLIRPass
    MLIRRegisterAllDialects
    MLIRRewrite
    MLIRSupport
    MLIRTableGen
    MLIRTblgenLib
    MLIRTranslateLib
)

add_executable(rt-run tools/rt-run.cpp)

target_link_libraries(rt-run RuntimeMLIR)

add_custom_command(TARGET rt-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_kernels/build/Release/cuda_kernels.dll"
        $<TARGET_FILE_DIR:rt-run>
    COMMENT "Copying cuda_kernels.dll to output directory"
)
