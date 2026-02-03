#pragma once
#include "core/tensor.h"
#define MAX_ELEMENTS 8

void dump_tensor(const Tensor& t,
                 const char* name,
                 int max_elements = MAX_ELEMENTS);

