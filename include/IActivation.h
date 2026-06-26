#pragma once
#include <string>

// Interfaz abstracta para funciones de activacion en GPU
class IActivation{
public:
    virtual ~IActivation() = default;

    // Propagacion hacia adelante in-place en la GPU
    virtual void forward(float* d_inout, int N, int total) = 0;

    // Propagacion hacia atras (backward) in-place en la GPU
    virtual void backward(float* d_grad,const float* d_fwd_output,int N, int total) = 0;

    // Nombre de la activacion
    virtual std::string name() const = 0;
};
