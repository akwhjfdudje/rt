#include "core/graph.h"
#include "core/op.h"

void Graph::add(Op* op) {
    ops.push_back(op);
}

void Graph::execute() {
    for (Op* op : ops) {
        op->run();
    }
}

void Graph::clear() {
    ops.clear();
}

