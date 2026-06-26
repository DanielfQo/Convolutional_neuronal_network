#pragma once

#include "Tensor.h"
#include <string>

// Interfaz para las funciones de perdida (Loss)
class ILoss {
public:
    virtual ~ILoss() = default;

    // Calcula el valor escalar de la perdida
    virtual float compute(const Tensor& predictions, const Tensor& labels) = 0;

    // Calcula el gradiente de la perdida respecto a las predicciones
    virtual Tensor gradient(const Tensor& predictions, const Tensor& labels) = 0;

    // Nombre legible de la funcion de perdida
    virtual std::string name() const = 0;
};
