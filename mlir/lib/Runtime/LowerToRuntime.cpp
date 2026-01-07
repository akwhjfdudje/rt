#include "Runtime/RuntimeOps.h"
#include "Runtime/RuntimeDialect.h"
#include "runtime_api.h"
#include "debug/trace.h"
#include "debug/snapshot.h"

using namespace mlir;

void lowerRtNoiseOp(Tensor* out, int seed) {
    trace_begin("rt-noise");
    dump_tensor(*out, "rt-noise:input");
    debug_step("before rt-noise");

    rt_noise(out, seed);

    dump_tensor(*out, "rt-noise:output");
    guard_nan_inf(*out, "rt-noise");
    trace_end("rt-noise");
}

