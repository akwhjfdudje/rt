#include "debug/trace.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <mutex>

struct TraceEvent {
    std::string name;
    float ms;
};

static std::vector<TraceEvent> g_events;
static cudaEvent_t g_start, g_end;
static std::mutex g_mutex;

void trace_begin(const char* kernel_name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_end);
    cudaEventRecord(g_start);
}

void trace_end(const char* kernel_name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    cudaEventRecord(g_end);
    cudaEventSynchronize(g_end);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, g_start, g_end);

    g_events.push_back({kernel_name, ms});

    cudaEventDestroy(g_start);
    cudaEventDestroy(g_end);
}

void trace_flush() {
    printf("\n=== Kernel Trace ===\n");
    for (size_t i = 0; i < g_events.size(); ++i) {
        printf("[%zu] %-20s : %6.3f ms\n",
               i,
               g_events[i].name.c_str(),
               g_events[i].ms);
    }
    g_events.clear();
}

