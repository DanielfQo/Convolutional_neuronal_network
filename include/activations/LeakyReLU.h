#pragma once
#include "IActivation.h"

// Activacion LeakyReLU
class LeakyReLU : public IActivation {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha_(alpha) {}
    void forward(float* d_inout, int N, int total) override;
    void backward(float* d_grad, const float* d_fwd_output, int N, int total) override;
    std::string name() const override { return "leaky_relu"; }
private:
    float alpha_;
};
