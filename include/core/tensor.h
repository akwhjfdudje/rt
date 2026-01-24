#pragma once
#include <vector>
#include <cstddef>
#include <cassert>

enum class DeviceType {
    CPU,
    CUDA
};

// TODO: make this make sense, it doesn't right now.
struct Tensor {
    void* device = nullptr;   // CUDA device pointer
    void* host   = nullptr;   // Optional host mirror
    std::vector<int64_t> shape;
    size_t bytes = 0;
    DeviceType device_type = DeviceType::CUDA;

    size_t numel() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }
};

