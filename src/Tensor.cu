#include "Tensor.h"
#include "CudaUtils.h"

#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>
#include <fstream>

// Kernel de CUDA para llenar datos en paralelo
__global__ static void fill_kernel(float* data, float val, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

// Reserva memoria en GPU
void Tensor::allocGPU(){
    int n = size();
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_data_, n * sizeof(float)));
    }
}

// Libera memoria en GPU y CPU
void Tensor::freeAll(){
    if (d_data_) { CUDA_CHECK(cudaFree(d_data_)); d_data_ = nullptr; }
    if (h_data_) { delete[] h_data_; h_data_ = nullptr; }
    N_ = C_ = H_ = W_ = 0;
}

// Constructor con dimensiones
Tensor::Tensor(int N, int C, int H, int W): N_(N), C_(C), H_(H), W_(W){
    allocGPU();
}

// Destructor
Tensor::~Tensor(){
    freeAll();
}

// Constructor de movimiento
Tensor::Tensor(Tensor&& other) noexcept : d_data_(other.d_data_), h_data_(other.h_data_),
      N_(other.N_), C_(other.C_), H_(other.H_), W_(other.W_){
    other.d_data_ = nullptr;
    other.h_data_ = nullptr;
    other.N_ = other.C_ = other.H_ = other.W_ = 0;
}

// Operador de asignacion por movimiento
Tensor& Tensor::operator=(Tensor&& other) noexcept{
    if (this != &other) {
        freeAll();
        d_data_ = other.d_data_; h_data_ = other.h_data_;
        N_ = other.N_; C_ = other.C_; H_ = other.H_; W_ = other.W_;
        other.d_data_ = nullptr; other.h_data_ = nullptr;
        other.N_ = other.C_ = other.H_ = other.W_ = 0;
    }
    return *this;
}

// Duplicar el tensor en la GPU
Tensor Tensor::clone() const{
    Tensor out(N_, C_, H_, W_);
    if (size() > 0 && d_data_) {
        CUDA_CHECK(cudaMemcpy(out.d_data_, d_data_, size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return out;
}

// Obtener datos en la CPU
float* Tensor::cpu(){
    toCPU();
    return h_data_;
}

const float* Tensor::cpu() const{
    toCPU();
    return h_data_;
}

// Llenar con ceros
void Tensor::zeros(){
    if (size() > 0 && d_data_) {
        CUDA_CHECK(cudaMemset(d_data_, 0, size() * sizeof(float)));
    }
}

// Llenar con un valor constante
void Tensor::fill(float val){
    if (size() > 0 && d_data_) {
        fill_kernel<<<gridSize(size()), BLOCK_SIZE>>>(d_data_, val, size());
        CUDA_CHECK(cudaGetLastError());
    }
}

// Llenar usando una distribucion normal
void Tensor::randomNormal(float mean, float std)
{
    int n = size();
    if (n == 0) return;

    if (!h_data_) h_data_ = new float[n];

    std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<float> dist(mean, std);
    for (int i = 0; i < n; ++i) h_data_[i] = dist(rng);

    CUDA_CHECK(cudaMemcpy(d_data_, h_data_,
                          n * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// Subir datos a la GPU
void Tensor::toGPU()
{
    if (!h_data_ || size() == 0) return;
    CUDA_CHECK(cudaMemcpy(d_data_, h_data_,
                          size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// Descargar datos a la CPU
void Tensor::toCPU() const{
    if (size() == 0 || !d_data_) return;
    if (!h_data_) h_data_ = new float[size()];
    CUDA_CHECK(cudaMemcpy(h_data_, d_data_, size() * sizeof(float), cudaMemcpyDeviceToHost));
}

// Guardar en archivo
void Tensor::save(std::ofstream& f) const{
    f.write(reinterpret_cast<const char*>(&N_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&C_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&H_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&W_), sizeof(int));
    if (size() > 0) {
        toCPU();
        f.write(reinterpret_cast<const char*>(h_data_), size() * sizeof(float));
    }
}

// Cargar desde archivo
void Tensor::load(std::ifstream& f){
    freeAll();
    f.read(reinterpret_cast<char*>(&N_), sizeof(int));
    f.read(reinterpret_cast<char*>(&C_), sizeof(int));
    f.read(reinterpret_cast<char*>(&H_), sizeof(int));
    f.read(reinterpret_cast<char*>(&W_), sizeof(int));
    allocGPU();
    if (size() > 0) {
        h_data_ = new float[size()];
        f.read(reinterpret_cast<char*>(h_data_), size() * sizeof(float));
        toGPU();
    }
}
