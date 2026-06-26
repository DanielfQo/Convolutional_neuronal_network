#pragma once
#include "ILoss.h"

// Funcion de perdida de Entropia Cruzada
class CrossEntropyLoss : public ILoss {
public:
    float compute(const Tensor& predictions, const Tensor& labels) override;
    Tensor gradient(const Tensor& predictions, const Tensor& labels) override;
    std::string name() const override { return "cross_entropy"; }
};
