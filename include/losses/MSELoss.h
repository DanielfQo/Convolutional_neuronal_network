#pragma once
#include "ILoss.h"

// Funcion de perdida de Error Cuadratico Medio (MSE)
class MSELoss : public ILoss {
public:
    float compute(const Tensor& predictions, const Tensor& labels) override;
    Tensor gradient(const Tensor& predictions, const Tensor& labels) override;
    std::string name() const override { return "mse"; }
};
