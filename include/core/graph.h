#pragma once
#include <vector>

class Op;

class Graph {
public:
    void add(Op* op);
    void execute();
    void clear();

private:
    std::vector<Op*> ops;
};

