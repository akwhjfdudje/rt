#include "debug/step.h"
#include <cstdio>

static bool g_step_enabled = false;

void debug_enable_steps(bool enable) {
    g_step_enabled = enable;
}

void debug_step(const char* label) {
    if (!g_step_enabled) return;

    printf("\n[step] %s — press ENTER to continue...\n", label);
    getchar();
}

