#pragma once
#include "core/tensor.h"

float diff_l2(const Tensor& gpu, const Tensor& cpu);
bool diff_check(const Tensor& gpu,
                const Tensor& cpu,
                float tolerance,
                const char* label);

