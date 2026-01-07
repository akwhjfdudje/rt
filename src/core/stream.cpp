#include "core/stream.h"
#include <stdexcept>

Stream::Stream() {
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed");
    }
}

Stream::~Stream() {
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

