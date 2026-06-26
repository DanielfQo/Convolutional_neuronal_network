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

#include <memory>
#include <cstdio>
#include <vector>

// Registra todas las activaciones y perdidas en sus respectivas fabricas
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

// Construye una red neuronal con arquitectura LeNet-5
static Network buildLeNet5() {
    auto& af = ActivationFactory::instance();
    auto& lf = LossFactory::instance();

    Network net;
    net.setBatchSize(32);
    net.setLoss(lf.create("cross_entropy"));

    // Entrada: 1 canal de 28x28
    net.addLayer(std::make_unique<ConvLayer>(1,  6, 5, 1, 0, af.create("relu")));
    net.addLayer(std::make_unique<PoolLayer>(2, 2, PoolType::MAX));
    net.addLayer(std::make_unique<ConvLayer>(6, 16, 5, 1, 0, af.create("relu")));
    net.addLayer(std::make_unique<PoolLayer>(2, 2, PoolType::MAX));
    net.addLayer(std::make_unique<FCLayer>(16 * 4 * 4, 120, af.create("relu")));
    net.addLayer(std::make_unique<FCLayer>(120, 84, af.create("tanh")));
    net.addLayer(std::make_unique<FCLayer>(84, 10, af.create("softmax")));

    return net;
}


// Punto de entrada del programa
int main() {
    initCuda();

    return 0;
}
