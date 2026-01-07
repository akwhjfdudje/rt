#include "debug/diff.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>

float diff_l2(const Tensor& gpu, const Tensor& cpu) {
    assert(gpu.bytes == cpu.bytes);

    int count = gpu.bytes / sizeof(float);
    std::vector<float> g(count);

    cudaMemcpy(g.data(), gpu.device, gpu.bytes, cudaMemcpyDeviceToHost);

    float acc = 0.0f;
    for (int i = 0; i < count; ++i) {
        float d = g[i] - static_cast<float*>(cpu.host)[i];
        acc += d * d;
    }
    return std::sqrt(acc / count);
}

bool diff_check(const Tensor& gpu,
                const Tensor& cpu,
                float tolerance,
                const char* label) {
    float err = diff_l2(gpu, cpu);
    if (err > tolerance) {
        printf("[diff] %s FAILED (L2 = %f, tol = %f)\n",
               label, err, tolerance);
        return false;
    }
    return true;
}

