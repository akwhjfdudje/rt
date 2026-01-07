#include "runtime_api.h"
#include "core/allocator.h"
#include <cuda_runtime.h>

static Allocator g_allocator;

extern "C" {

Tensor* rt_alloc(int rank, const int* shape, size_t element_size) {
    std::vector<int> dims(shape, shape + rank);
    Tensor* t = new Tensor(
        g_allocator.allocate(dims, element_size, DeviceType::CUDA)
    );
    return t;
}

void rt_free(Tensor* t) {
    if (!t) return;
    g_allocator.free(*t);
    delete t;
}

void rt_sync() {
    cudaDeviceSynchronize();
}

}

