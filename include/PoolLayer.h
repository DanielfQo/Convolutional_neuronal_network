#pragma once

#include "Layer.h"
#include <memory>

// Tipos de agrupamiento (pooling)
enum class PoolType { MAX, AVERAGE };

// Capa de pooling (agrupamiento)
class PoolLayer : public Layer {
public:
    // Constructor de la capa de pooling
    PoolLayer(int pool_size, int stride = -1, PoolType pool_type = PoolType::MAX);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    void updateWeights(float /*lr*/) override {}

    void save(std::ofstream& f) const override;
    void load(std::ifstream& f) override;
    std::string summary() const override;

private:
    PoolType pool_type_;
    int pool_size_;
    int stride_;

    // Mascara para guardar los indices del maximo valor en Max Pooling
    std::unique_ptr<Tensor> mask_;

    int out_H_ = 0, out_W_ = 0;
};
