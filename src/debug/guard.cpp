#include "debug/guard.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>

bool guard_nan_inf(const Tensor& t, const char* label) {
    if (!t.device || t.bytes == 0) return true;

    int count = t.bytes / sizeof(float);
    std::vector<float> host(count);

    cudaMemcpy(host.data(), t.device, t.bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i) {
        if (std::isnan(host[i]) || std::isinf(host[i])) {
            printf("[guard] %s: INVALID VALUE at index %d (%f)\n",
                   label, i, host[i]);
            return false;
        }
    }
    return true;
}

