#pragma once

#include "CudaUtils.h"
#include <fstream>
#include <cstring> 

/*
 *Clase Tensor en 4D (N, C, H, W) para almacenar y gestionar datos en GPU y CPU.
*/
class Tensor {
public:
    
    /// Constructor por defecto (crea un tensor vacio con dimensiones en 0).
    Tensor() = default;
    
    /// Constructor que define las dimensiones del tensor y reserva memoria en GPU.
    Tensor(int N, int C, int H, int W);
    
    /// Destructor. Libera todos los recursos asociados (GPU y CPU).
    ~Tensor();

    // Se deshabilita la copia convencional para evitar duplicados accidentales de memoria GPU.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    /// Constructor de movimiento (transfiere la propiedad de los recursos).
    Tensor(Tensor&& other) noexcept;
    
    /// Operador de asignacion por movimiento.
    Tensor& operator=(Tensor&& other) noexcept;

    /// Realiza una copia profunda (duplica memoria en GPU).
    Tensor clone() const;

    /// Retorna el puntero crudo a los datos en la memoria de GPU.
    float* gpu()  const { return d_data_; }

    /// Retorna el puntero a los datos en CPU, sincronizandolos primero desde la GPU.
    float* cpu();
    
    /// Retorna el puntero de solo lectura a los datos en CPU, sincronizandolos primero.
    const float* cpu() const;

    int N() const { return N_; } ///< Tamaño del lote (Batch size).
    int C() const { return C_; } ///< Cantidad de canales (Channels).
    int H() const { return H_; } ///< Alto (Height).
    int W() const { return W_; } ///< Ancho (Width).

    /// Retorna el numero total de elementos flotantes en el tensor (N * C * H * W).
    int size() const { return N_ * C_ * H_ * W_; }

    /// Retorna el tamaño del tensor expresado en bytes.
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(float); }
    
    /// Indica si el tensor no tiene memoria reservada en GPU.
    bool empty() const { return d_data_ == nullptr; }

    
    /// Llena el tensor con ceros en la GPU.
    void zeros();
    
    /// Llena el tensor con un valor constante en la GPU.
    void fill(float val);

    /// Llena el tensor usando una distribucion normal en CPU y luego sube los datos a la GPU.
    void randomNormal(float mean = 0.0f, float std = 1.0f);

    /// Copia los datos desde la CPU (Host) hacia la GPU (Device).
    void toGPU();
    
    /// Copia los datos desde la GPU (Device) hacia la CPU (Host) (asigna memoria en CPU si es necesario).
    void toCPU() const;

    /// Guarda el tensor en disco en formato binario.
    void save(std::ofstream& f) const;

    /// Carga el tensor desde disco en formato binario y lo sube a la GPU.
    void load(std::ifstream& f);

private:
    float* d_data_ = nullptr;          ///< Puntero a la memoria en GPU (Device).
    mutable float* h_data_ = nullptr;  ///< Puntero a la memoria en CPU (Host), asignado de forma perezosa.
    int N_ = 0, C_ = 0, H_ = 0, W_ = 0; ///< Dimensiones del tensor.

    /// Reserva de forma interna memoria en GPU.
    void allocGPU();
    
    /// Libera toda la memoria (tanto de GPU como de CPU) y limpia las dimensiones.
    void freeAll();
};
