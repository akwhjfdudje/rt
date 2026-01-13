#pragma once
#include "core/tensor.h"

extern "C" {

// Allocate a CUDA tensor
Tensor* rt_alloc(int rank, const int64_t* shape, size_t element_size);

// Free tensor
void rt_free(Tensor* t);

// Synchronize device
void rt_sync();

}

