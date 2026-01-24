#include "core/allocator.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

Tensor Allocator::allocate(const std::vector<int64_t>& shape,
                           size_t element_size,
                           DeviceType device) {
    Tensor t;
    t.shape = shape;
    t.device_type = device;
    t.bytes = element_size;
    for (size_t d : shape) t.bytes *= d;

    if (device == DeviceType::CUDA) {
        cudaError_t err = cudaMalloc(&t.device, t.bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    } else {
        t.host = std::malloc(t.bytes);
        if (!t.host) {
            throw std::runtime_error("malloc failed");
        }
    }

    return t;
}

void Allocator::free(Tensor& t) {
    if (t.device && t.device_type == DeviceType::CUDA) {
        cudaFree(t.device);
    }
    if (t.host && t.device_type == DeviceType::CPU) {
        std::free(t.host);
    }
    t.device = nullptr;
    t.host = nullptr;
    t.bytes = 0;
}

void Allocator::fill(Tensor& t, std::vector<int64_t>& filler) {
    size_t size = t.numel();
}
