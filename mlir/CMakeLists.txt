add_library(RuntimeMLIR STATIC
    lib/Runtime/RuntimeDialect.cpp
    lib/Runtime/Passes.cpp
)

target_include_directories(RuntimeMLIR PUBLIC
    include
    ../adapters/include
)

target_link_libraries(RuntimeMLIR PUBLIC
    runtime
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
add_executable(rt-debug tools/rt-debug.cpp)
add_executable(rt-graph tools/rt-graph.cpp)

target_link_libraries(rt-run RuntimeMLIR)
target_link_libraries(rt-debug RuntimeMLIR)
target_link_libraries(rt-graph RuntimeMLIR)

add_custom_command(TARGET rt-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_kernels/build/Release/cuda_kernels.dll"
        $<TARGET_FILE_DIR:rt-run>
    COMMENT "Copying cuda_kernels.dll to output directory"
)
