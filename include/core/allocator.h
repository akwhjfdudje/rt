#pragma once
#include "tensor.h"

class Allocator {
public:
    Tensor allocate(const std::vector<int>& shape,
                    size_t element_size,
                    DeviceType device = DeviceType::CUDA);

    void free(Tensor& t);
};

