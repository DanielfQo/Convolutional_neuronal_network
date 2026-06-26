#pragma once

#include "IActivation.h"
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <unordered_map>

// Fabrica singleton para crear activaciones por nombre
class ActivationFactory
{
public:
    using Creator = std::function<std::unique_ptr<IActivation>()>;

    static ActivationFactory& instance(){
        static ActivationFactory inst;
        return inst;
    }

    // Registra un creador de activacion bajo un nombre especifico
    void registerActivation(const std::string& name, Creator creator){
        registry_[name] = std::move(creator);
    }

    // Crea una instancia de activacion por su nombre
    std::unique_ptr<IActivation> create(const std::string& name) const{
        auto it = registry_.find(name);
        if (it == registry_.end()) {
            throw std::runtime_error(
                "[ActivationFactory] Activacion desconocida: \"" + name + "\".");
        }
        return it->second();
    }

    // Verifica si la activacion esta registrada
    bool has(const std::string& name) const{
        return registry_.count(name) > 0;
    }

    // Limpia el registro (util para testeo)
    void clear(){
        registry_.clear();
    }

private:
    ActivationFactory() = default;
    std::unordered_map<std::string, Creator> registry_;
};
