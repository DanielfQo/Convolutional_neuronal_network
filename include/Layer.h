#pragma once

#include "Tensor.h"
#include "IActivation.h"

#include <memory>
#include <string>
#include <fstream>

// Tipos de capa para serializacion
enum class LayerType { CONV = 0, POOL = 1, FC = 2 };

// Clase base abstracta para todas las capas de la red
class Layer
{
public:
    virtual ~Layer() = default;

    // Propagacion hacia adelante
    virtual Tensor forward(const Tensor& input) = 0;

    // Propagacion hacia atras
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Actualizacion de pesos con SGD
    virtual void updateWeights(float lr) = 0;

    // Guardar y cargar pesos
    virtual void save(std::ofstream& f) const = 0;
    virtual void load(std::ifstream& f) = 0;

    // Resumen de la capa
    virtual std::string summary() const = 0;
    LayerType type() const { return type_; }

protected:
    LayerType type_{LayerType::FC};
    std::unique_ptr<IActivation> activation_{nullptr};
    Tensor output_cache_{};
    Tensor input_cache_{};

    // Auxiliares de serializacion
    static void writeStr(std::ofstream& f, const std::string& s);
    static std::string readStr(std::ifstream& f);
};
