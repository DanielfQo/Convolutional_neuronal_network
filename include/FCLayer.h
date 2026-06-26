#pragma once

#include "Layer.h"
#include <memory>

// Capa completamente conectada (Densa)
class FCLayer : public Layer {
public:
    // Constructor de la capa completamente conectada
    FCLayer(int input_size, int output_size,
            std::unique_ptr<IActivation> act = nullptr);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void   updateWeights(float lr) override;
    void   save(std::ofstream& f) const override;
    void   load(std::ifstream& f) override;
    std::string summary() const override;

private:
    Tensor weights_;
    Tensor bias_;
    Tensor grad_w_;
    Tensor grad_b_;

    int in_size_, out_size_;
};
