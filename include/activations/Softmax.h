#pragma once
#include "IActivation.h"

// Activacion Softmax estable
class Softmax : public IActivation {
public:
    void forward(float* d_inout, int N, int total) override;
    void backward(float* d_grad, const float* d_fwd_output, int N, int total) override;
    std::string name() const override { return "softmax"; }
};
