#include "Network.h"
#include "CudaUtils.h"

#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <numeric>

// Constructor.
Network::Network() {}

// Destructor.
Network::~Network() {}

// Constructor de movimiento
Network::Network(Network&& other) noexcept
    : layers_(std::move(other.layers_)),
      loss_fn_(std::move(other.loss_fn_)),
      batch_size_(other.batch_size_) {}

// Operador de asignacion por movimiento
Network& Network::operator=(Network&& other) noexcept {
    if (this != &other) {
        layers_ = std::move(other.layers_);
        loss_fn_ = std::move(other.loss_fn_);
        batch_size_ = other.batch_size_;
    }
    return *this;
}

// Agrega una capa a la red.
void Network::addLayer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

// Define la funcion de perdida.
void Network::setLoss(std::unique_ptr<ILoss> loss) {
    loss_fn_ = std::move(loss);
}

// Propagacion hacia adelante secuencial.
Tensor Network::forward(const Tensor& input) {
    Tensor current(input.N(), input.C(), input.H(), input.W());
    CUDA_CHECK(cudaMemcpy(current.gpu(), input.gpu(),
                          input.bytes(), cudaMemcpyDeviceToDevice));

    for (auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

// Inferencia o prediccion.
Tensor Network::predict(const Tensor& input) {
    return forward(input);
}

// Propagacion hacia atras secuencial (en orden inverso).
void Network::backward(const Tensor& loss_grad) {
    Tensor grad(loss_grad.N(), loss_grad.C(), loss_grad.H(), loss_grad.W());
    CUDA_CHECK(cudaMemcpy(grad.gpu(), loss_grad.gpu(), loss_grad.bytes(), cudaMemcpyDeviceToDevice));

    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad = layers_[i]->backward(grad);
    }
}

// Actualiza los pesos de todas las capas.
void Network::updateWeights(float lr) {
    for (auto& layer : layers_) layer->updateWeights(lr);
}

// Ciclo de entrenamiento principal por epocas y lotes.
void Network::train(const std::vector<Tensor>& data,
                    const std::vector<Tensor>& labels,
                    int epochs, float lr) {
    if (!loss_fn_) throw std::runtime_error("Network::train - no se definio funcion de perdida");
    if (data.size() != labels.size())
        throw std::runtime_error("Network::train - discrepancia en tamano de datos/etiquetas");

    int num_samples = static_cast<int>(data.size());
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.f;
        int num_batches = 0;

        for (int start = 0; start < num_samples; start += batch_size_) {
            int end = std::min(start + batch_size_, num_samples);
            int batch_n = end - start;

            int C = data[0].C(), H = data[0].H(), W = data[0].W();
            int cls = labels[0].size();

            Tensor batch_x(batch_n, C, H, W);
            Tensor batch_y(batch_n, 1, 1, cls);
            batch_x.zeros();
            batch_y.zeros();

            for (int b = 0; b < batch_n; ++b) {
                int idx = indices[start + b];
                CUDA_CHECK(cudaMemcpy(
                    batch_x.gpu() + b * C * H * W,
                    data[idx].gpu(),
                    data[idx].bytes(),
                    cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(
                    batch_y.gpu() + b * cls,
                    labels[idx].gpu(),
                    labels[idx].bytes(),
                    cudaMemcpyDeviceToDevice));
            }

            Tensor output = forward(batch_x);

            float loss = loss_fn_->compute(output, batch_y);
            epoch_loss += loss;

            Tensor grad = loss_fn_->gradient(output, batch_y);
            backward(grad);
            updateWeights(lr);
            ++num_batches;
        }

        printf("Epoch %3d/%d - loss: %.6f\n",
               epoch + 1, epochs, epoch_loss / num_batches);
    }
}

static const char MAGIC[4] = {'C', 'N', 'N', '1'};

// Guarda el modelo en un archivo binario.
void Network::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Network::save - no se pudo abrir: " + path);

    f.write(MAGIC, 4);
    int n = static_cast<int>(layers_.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(int));

    for (const auto& layer : layers_) {
        int t = static_cast<int>(layer->type());
        f.write(reinterpret_cast<const char*>(&t), sizeof(int));
        layer->save(f);
    }
    printf("Modelo guardado en '%s'\n", path.c_str());
}

// Carga el modelo desde un archivo binario.
void Network::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Network::load - no se pudo abrir: " + path);

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0)
        throw std::runtime_error("Network::load - formato de archivo invalido");

    int n;
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (n != static_cast<int>(layers_.size()))
        throw std::runtime_error("Network::load - discrepancia en conteo de capas");

    for (auto& layer : layers_) {
        int t;
        f.read(reinterpret_cast<char*>(&t), sizeof(int));
        if (t != static_cast<int>(layer->type()))
            throw std::runtime_error("Network::load - discrepancia en tipo de capa");
        layer->load(f);
    }
    printf("Modelo cargado desde '%s'\n", path.c_str());
}

// Muestra un resumen de la arquitectura por consola.
void Network::summary() const {
    printf("\n###########################################################\n");
    printf("                   Network Architecture                      \n");
    printf("###########################################################\\n");
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        std::string s = layers_[i]->summary();
        printf("[%2d] %s\n", i + 1, s.c_str());
    }
    printf("###########################################################\\n");
    printf("  Loss : %-52s\n",
           loss_fn_ ? loss_fn_->name().c_str() : "none");
    printf("  Batch: %-52d\n", batch_size_);
    printf("###########################################################\\n");
}
