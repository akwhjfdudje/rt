#pragma once
#include "core/tensor.h"

void rt_generateNoise(Tensor* output, float min_val = -1.0f, float max_val = 1.0f, unsigned int seed = 42u);
