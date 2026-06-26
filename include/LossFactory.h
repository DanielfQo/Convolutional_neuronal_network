#pragma once

#include "ILoss.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>

// Fabrica singleton para crear funciones de perdida por nombre
class LossFactory {
public:
    using Creator = std::function<std::unique_ptr<ILoss>()>;

    static LossFactory& instance() {
        static LossFactory inst;
        return inst;
    }

    // Registra una funcion de perdida bajo un nombre
    void registerLoss(const std::string& name, Creator creator) {
        registry_[name] = std::move(creator);
    }

    // Crea una funcion de perdida por su nombre
    std::unique_ptr<ILoss> create(const std::string& name) const {
        auto it = registry_.find(name);
        if (it == registry_.end())
            throw std::runtime_error("LossFactory: perdida desconocida '" + name + "'");
        return it->second();
    }

    // Verifica si la funcion de perdida esta registrada
    bool has(const std::string& name) const {
        return registry_.count(name) > 0;
    }

private:
    LossFactory() = default;
    std::unordered_map<std::string, Creator> registry_;
};
