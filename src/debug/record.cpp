#include "debug/record.h"
#include <fstream>
#include <cstdio>

static std::vector<KernelRecord> g_records;

void record_begin() {
    g_records.clear();
}

void record_kernel(const KernelRecord& r) {
    g_records.push_back(r);
}

void record_end(const char* path) {
    std::ofstream out(path);
    for (auto& r : g_records) {
        out << r.name;
        for (int s : r.shapes) out << " " << s;
        for (int p : r.params) out << " " << p;
        out << "\n";
    }
    printf("[record] wrote %zu kernels to %s\n",
           g_records.size(), path);
}

void replay(const char* path) {
    printf("[replay] loading %s\n", path);
    // v1: just show; v2: re-execute adapters
}

