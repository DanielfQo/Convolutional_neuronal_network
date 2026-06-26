#include "CudaUtils.h"
#include "Network.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "FCLayer.h"
#include "ActivationFactory.h"
#include "LossFactory.h"

#include "activations/ReLU.h"
#include "activations/Sigmoid.h"
#include "activations/Tanh.h"
#include "activations/LeakyReLU.h"
#include "activations/Softmax.h"
#include "activations/Linear.h"

#include "losses/CrossEntropyLoss.h"
#include "losses/MSELoss.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>

// Registra todas las activaciones y perdidas en las fabricas
static void registerAll() {
    auto& af = ActivationFactory::instance();
    af.registerActivation("relu",      []{ return std::make_unique<ReLU>(); });
    af.registerActivation("sigmoid",   []{ return std::make_unique<Sigmoid>(); });
    af.registerActivation("tanh",      []{ return std::make_unique<Tanh>(); });
    af.registerActivation("softmax",   []{ return std::make_unique<Softmax>(); });
    af.registerActivation("leakyrelu", []{ return std::make_unique<LeakyReLU>(0.01f); });
    af.registerActivation("linear",    []{ return std::make_unique<Linear>(); });

    auto& lf = LossFactory::instance();
    lf.registerLoss("cross_entropy", []{ return std::make_unique<CrossEntropyLoss>(); });
    lf.registerLoss("mse",           []{ return std::make_unique<MSELoss>(); });
}

bool loadBloodMNIST(const std::string& filepath, std::vector<Tensor>& x_data, std::vector<Tensor>& y_data, int num_classes = 8) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) {
        std::cerr << "Error al abrir el archivo: " << filepath << std::endl;
        return false;
    }

    int nSamples = 0, width = 0, height = 0, channels = 0;
    f.read(reinterpret_cast<char*>(&nSamples), sizeof(int));
    f.read(reinterpret_cast<char*>(&width), sizeof(int));
    f.read(reinterpret_cast<char*>(&height), sizeof(int));
    f.read(reinterpret_cast<char*>(&channels), sizeof(int));

    std::cout << "Cargando " << filepath << ": " << nSamples << " muestras, "
              << width << "x" << height << "x" << channels << std::endl;

    x_data.reserve(nSamples);
    y_data.reserve(nSamples);

    int img_size = width * height * channels;
    std::vector<float> h_pixels(img_size);

    for (int i = 0; i < nSamples; ++i) {
        int label = 0;
        f.read(reinterpret_cast<char*>(&label), sizeof(int));
        f.read(reinterpret_cast<char*>(h_pixels.data()), img_size * sizeof(float));

        // Tensor de imagen: shape (1, channels, height, width)
        Tensor img(1, channels, height, width);
        std::memcpy(img.cpu(), h_pixels.data(), img_size * sizeof(float));
        img.toGPU();
        x_data.push_back(std::move(img));

        // Tensor de etiqueta (One-hot vector): shape (1, 1, 1, num_classes)
        Tensor one_hot(1, 1, 1, num_classes);
        float* oh_ptr = one_hot.cpu();
        std::memset(oh_ptr, 0, num_classes * sizeof(float));
        if (label >= 0 && label < num_classes) {
            oh_ptr[label] = 1.0f;
        }
        one_hot.toGPU();
        y_data.push_back(std::move(one_hot));
    }

    return true;
}

int main() {
    initCuda();
    registerAll();

    std::vector<Tensor> x_train, y_train;
    std::vector<Tensor> x_val, y_val;
    std::vector<Tensor> x_test, y_test;

    if (!loadBloodMNIST("data/bloodmnist_train.bin", x_train, y_train) ||
        !loadBloodMNIST("data/bloodmnist_val.bin", x_val, y_val) ||
        !loadBloodMNIST("data/bloodmnist_test.bin", x_test, y_test)) {
        std::cerr << "Error cargando los datos de BloodMNIST." << std::endl;
        return 1;
    }

    auto& af = ActivationFactory::instance();
    auto& lf = LossFactory::instance();

    Network net;
    net.setBatchSize(64);
    net.setLoss(lf.create("cross_entropy"));

    // Entrada: 3 canales (RGB) de 28x28
    net.addLayer(std::make_unique<ConvLayer>(3,  6, 5, 1, 0, af.create("relu")));
    net.addLayer(std::make_unique<PoolLayer>(2, 2, PoolType::MAX));
    net.addLayer(std::make_unique<ConvLayer>(6, 16, 5, 1, 0, af.create("relu")));
    net.addLayer(std::make_unique<PoolLayer>(2, 2, PoolType::MAX));
    net.addLayer(std::make_unique<FCLayer>(16 * 4 * 4, 120, af.create("relu")));
    net.addLayer(std::make_unique<FCLayer>(120, 84, af.create("sigmoid")));
    net.addLayer(std::make_unique<FCLayer>(84, 8, af.create("softmax"))); 

    net.summary();

    // Entrenar por 3 
    std::cout << "\nIniciando entrenamiento..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    net.train(x_train, y_train, 50, 0.01f);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Tiempo de entrenamiento: " << diff.count() << " s\n" << std::endl;

    // Guardar modelo entrenado
    net.save("bloodmnist_lenet.bin");

    cleanupCuda();
    return 0;
}
