#pragma once
#include <string>
#include <vector>

struct KernelRecord {
    std::string name;
    std::vector<int> shapes;
    std::vector<int> params;
};

void record_begin();
void record_kernel(const KernelRecord& r);
void record_end(const char* path);
void replay(const char* path);

