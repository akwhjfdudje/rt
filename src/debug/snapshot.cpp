#include "debug/snapshot.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <algorithm>

void dump_tensor(const Tensor& t,
                 const char* name,
                 int max_elements) {
    if (!t.device || t.bytes == 0) {
        printf("[dump] %s : <empty>\n", name);
        return;
    }

    int count = t.bytes / sizeof(float);
    int dump_n = std::min(count, max_elements);

    std::vector<float> host(count);
    cudaMemcpy(host.data(), t.device, t.bytes, cudaMemcpyDeviceToHost);

    printf("\n[dump] Tensor %s\n", name);
    printf("  shape = [");
    for (size_t i = 0; i < t.shape.size(); ++i) {
        printf("%d%s", t.shape[i], i + 1 < t.shape.size() ? "," : "");
    }
    printf("]\n");

    for (int i = 0; i < dump_n; ++i) {
        printf("  [%d] = %f\n", i, host[i]);
    }
}

