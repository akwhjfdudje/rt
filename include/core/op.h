#pragma once

class Op {
public:
    virtual ~Op() = default;
    virtual void run() = 0;
};

