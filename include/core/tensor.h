#pragma once
#include <vector>
#include <cstddef>
#include <cassert>

enum class DeviceType {
    CPU,
    CUDA
};

struct Tensor {
    void* device = nullptr;   // CUDA device pointer
    void* host   = nullptr;   // Optional host mirror
    std::vector<int> shape;
    size_t bytes = 0;
    DeviceType device_type = DeviceType::CUDA;

    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= d;
        return n;
    }
};

