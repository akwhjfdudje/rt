#pragma once
#include <cuda_runtime.h>

class Stream {
public:
    Stream();
    ~Stream();

    cudaStream_t get() const { return stream; }

private:
    cudaStream_t stream = nullptr;
};

