#include "FCLayer.h"
#include "CudaUtils.h"

#include <cmath>
#include <cstdio>

// Kernel para sumar sesgo (bias) en capa densa
__global__ void fc_add_bias_kernel(float* out, const float* bias, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) out[idx] += bias[idx % M];
}

// Kernel para calcular el gradiente del sesgo en capa densa
__global__ void fc_grad_bias_kernel(const float* grad, float* gb, int N, int M) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    float s = 0.f;
    for (int i = 0; i < N; ++i) s += grad[i * M + j];
    gb[j] = s;
}

// Kernel para actualizar pesos con SGD
__global__ void sgd_kernel_fc(float* w, const float* gw, int n, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * gw[i];
}

// Kernels personalizados de multiplicacion de matrices para FCLayer

__global__ void matmul_forward_fc_kernel(
    const float* __restrict__ input,      // [N, in_size]
    const float* __restrict__ weights,    // [out_size, in_size]
    float* __restrict__ output,            // [N, out_size]
    int N, int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_size;
    if (idx >= total) return;

    int b = idx / out_size;
    int i = idx % out_size;

    float sum = 0.0f;
    for (int j = 0; j < in_size; ++j) {
        sum += input[b * in_size + j] * weights[i * in_size + j];
    }
    output[idx] = sum;
}

__global__ void matmul_backward_w_fc_kernel(
    const float* __restrict__ input,      // [N, in_size]
    const float* __restrict__ grad,       // [N, out_size]
    float* __restrict__ grad_w,           // [out_size, in_size]
    int N, int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_size * in_size;
    if (idx >= total) return;

    int i = idx / in_size;
    int j = idx % in_size;

    float sum = 0.0f;
    for (int b = 0; b < N; ++b) {
        sum += grad[b * out_size + i] * input[b * in_size + j];
    }
    grad_w[idx] = sum;
}

__global__ void matmul_backward_in_fc_kernel(
    const float* __restrict__ weights,     // [out_size, in_size]
    const float* __restrict__ grad,        // [N, out_size]
    float* __restrict__ grad_input,        // [N, in_size]
    int N, int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * in_size;
    if (idx >= total) return;

    int b = idx / in_size;
    int j = idx % in_size;

    float sum = 0.0f;
    for (int i = 0; i < out_size; ++i) {
        sum += grad[b * out_size + i] * weights[i * in_size + j];
    }
    grad_input[idx] = sum;
}

// Constructor
FCLayer::FCLayer(int input_size, int output_size,
                 std::unique_ptr<IActivation> act)
    : in_size_(input_size), out_size_(output_size),
      weights_(1, 1, output_size, input_size),
      bias_(1, 1, 1, output_size),
      grad_w_(1, 1, output_size, input_size),
      grad_b_(1, 1, 1, output_size) {

    type_ = LayerType::FC;
    activation_ = std::move(act);

    float std = std::sqrt(1.f / static_cast<float>(input_size));
    weights_.randomNormal(0.f, std);
    bias_.zeros();
}

// Propagacion hacia adelante
Tensor FCLayer::forward(const Tensor& input) {
    int N = input.N();

    input_cache_ = Tensor(N, 1, 1, in_size_);
    CUDA_CHECK(cudaMemcpy(input_cache_.gpu(), input.gpu(),
                          input.bytes(), cudaMemcpyDeviceToDevice));

    Tensor output(N, 1, 1, out_size_);

    matmul_forward_fc_kernel<<<gridSize(N * out_size_), BLOCK_SIZE>>>(
        input.gpu(), weights_.gpu(), output.gpu(), N, in_size_, out_size_
    );
    CUDA_CHECK(cudaGetLastError());

    fc_add_bias_kernel<<<gridSize(N * out_size_), BLOCK_SIZE>>>(output.gpu(), bias_.gpu(), N, out_size_);
    CUDA_CHECK(cudaGetLastError());

    if (activation_) {
        activation_->forward(output.gpu(), N, output.size());
        output_cache_ = output.clone();
    }

    return output;
}

// Propagacion hacia atras
Tensor FCLayer::backward(const Tensor& grad_output) {
    int N = input_cache_.N();

    Tensor grad(N, 1, 1, out_size_);
    CUDA_CHECK(cudaMemcpy(grad.gpu(), grad_output.gpu(),grad_output.bytes(), cudaMemcpyDeviceToDevice));
    
    if (activation_) {
        activation_->backward(grad.gpu(), output_cache_.gpu(), N, grad.size());
    }

    matmul_backward_w_fc_kernel<<<gridSize(out_size_ * in_size_), BLOCK_SIZE>>>(
        input_cache_.gpu(), grad.gpu(), grad_w_.gpu(), N, in_size_, out_size_
    );
    CUDA_CHECK(cudaGetLastError());

    fc_grad_bias_kernel<<<gridSize(out_size_), BLOCK_SIZE>>>(grad.gpu(), grad_b_.gpu(), N, out_size_);
    CUDA_CHECK(cudaGetLastError());

    Tensor grad_input(N, 1, 1, in_size_);
    matmul_backward_in_fc_kernel<<<gridSize(N * in_size_), BLOCK_SIZE>>>(
        weights_.gpu(), grad.gpu(), grad_input.gpu(), N, in_size_, out_size_
    );
    CUDA_CHECK(cudaGetLastError());

    return grad_input;
}

// Actualizar pesos
void FCLayer::updateWeights(float lr) {
    sgd_kernel_fc<<<gridSize(weights_.size()), BLOCK_SIZE>>>(weights_.gpu(), grad_w_.gpu(), weights_.size(), lr);
    sgd_kernel_fc<<<gridSize(bias_.size()), BLOCK_SIZE>>>(bias_.gpu(), grad_b_.gpu(), bias_.size(), lr);
    CUDA_CHECK(cudaGetLastError());
}

// Guardar en archivo binario
void FCLayer::save(std::ofstream& f) const {
    f.write(reinterpret_cast<const char*>(&in_size_),  sizeof(int));
    f.write(reinterpret_cast<const char*>(&out_size_), sizeof(int));
    weights_.save(f);
    bias_.save(f);
}

// Cargar desde archivo binario
void FCLayer::load(std::ifstream& f) {
    f.read(reinterpret_cast<char*>(&in_size_),  sizeof(int));
    f.read(reinterpret_cast<char*>(&out_size_), sizeof(int));
    weights_.load(f);
    bias_.load(f);
}

// Resumen legible
std::string FCLayer::summary() const {
    char buf[256];
    int params = out_size_ * in_size_ + out_size_;
    std::snprintf(buf, sizeof(buf),
        "FC(%d -> %d)                     act=%-10s  params=%d",
        in_size_, out_size_,
        activation_ ? activation_->name().c_str() : "none",
        params);
    return std::string(buf);
}
