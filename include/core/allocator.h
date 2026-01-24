#pragma once
#include "tensor.h"

class Allocator {
public:
    static Tensor allocate(const std::vector<int64_t>& shape,
                    size_t element_size,
                    DeviceType device = DeviceType::CUDA);

    void free(Tensor& t);
    void fill(Tensor& t, std::vector<int64_t>& filler);
};

