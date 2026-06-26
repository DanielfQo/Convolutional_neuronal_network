#pragma once

#include "Layer.h"
#include "ILoss.h"

#include <memory>
#include <string>
#include <vector>

// Clase principal de la red neuronal secuencial
class Network {
public:
    Network();
    ~Network();

    // Deshabilitar copias
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    // Permitir movimientos
    Network(Network&& other) noexcept;
    Network& operator=(Network&& other) noexcept;

    // Agregar capas y funcion de perdida
    void addLayer(std::unique_ptr<Layer> layer);
    void setLoss(std::unique_ptr<ILoss> loss);

    // Definir tamano de lote (batch size)
    void setBatchSize(int batch_size) { batch_size_ = batch_size; }

    // Inferencia
    Tensor forward(const Tensor& input);
    Tensor predict(const Tensor& input);

    // Entrenamiento
    void backward(const Tensor& loss_grad);
    void updateWeights(float lr);

    // Ciclo de entrenamiento completo
    void train(const std::vector<Tensor>& data,
               const std::vector<Tensor>& labels,
               int epochs, float lr);

    // Guardar y cargar modelo en disco
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Resumen de la red en consola
    void summary() const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<ILoss> loss_fn_;
    int batch_size_ = 1;
};
