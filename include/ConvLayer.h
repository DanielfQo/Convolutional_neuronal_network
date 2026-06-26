#pragma once

#include "Layer.h"
#include "Tensor.h"
#include "IActivation.h"

#include <memory>
#include <string>
#include <fstream>

// Capa de convolucion 2D
class ConvLayer : public Layer{
public:
    // Constructor de la capa convolucional
    ConvLayer(int in_channels, int out_channels, int kernel_size,int stride = 1, int padding = 0,std::unique_ptr<IActivation> act = nullptr);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void updateWeights(float lr) override;

    void save(std::ofstream& f) const override;
    void load(std::ifstream& f) override;
    std::string summary() const override;

private:
    // Hiperparametros
    int in_ch_, out_ch_, ksize_, stride_, padding_;

    // Parametros entrenables
    Tensor weights_;
    Tensor bias_;
    Tensor grad_w_;
    Tensor grad_b_;

    // Buffer temporal im2col cacheado para el paso backward
    Tensor col_cache_;

    int out_H_ = 0, out_W_ = 0;

    // Funciones auxiliares
    void initWeights();
    void im2col(const Tensor& input, Tensor& col,int H_out, int W_out) const;
    void col2im(const Tensor& col, Tensor& grad_input,int H_in, int W_in, int H_out, int W_out) const;
    void addBias(Tensor& out, int H_out, int W_out) const;
    void biasGrad(const Tensor& grad, int H_out, int W_out);
};
