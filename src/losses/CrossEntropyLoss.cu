#include "losses/CrossEntropyLoss.h"
#include "CudaUtils.h"
#include <cmath>

// Kernel para forward de entropia cruzada en GPU
__global__ void ce_loss_kernel(const float* pred, const float* label,
                               float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -label[i] * logf(pred[i] + 1e-7f);
}

// Kernel para backward de entropia cruzada en GPU
__global__ void ce_grad_kernel(const float* pred, const float* label,
                               float* grad, int n, float inv_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = (pred[i] - label[i]) * inv_N;
}

// Funcion auxiliar para sumar valores de GPU en CPU
static float gpuSum(const float* d_buf, int n) {
    float* h = new float[n];
    cudaMemcpy(h, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += h[i];
    delete[] h;
    return s;
}

float CrossEntropyLoss::compute(const Tensor& predictions, const Tensor& labels) {
    int n = predictions.size();
    float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));
    ce_loss_kernel<<<gridSize(n), BLOCK_SIZE>>>(predictions.gpu(), labels.gpu(), d_tmp, n);
    CUDA_CHECK(cudaGetLastError());
    float loss = gpuSum(d_tmp, n) / static_cast<float>(predictions.N());
    CUDA_CHECK(cudaFree(d_tmp));
    return loss;
}

Tensor CrossEntropyLoss::gradient(const Tensor& predictions, const Tensor& labels) {
    int n = predictions.size();
    Tensor grad(predictions.N(), predictions.C(), predictions.H(), predictions.W());
    float inv_N = 1.f / static_cast<float>(predictions.N());
    ce_grad_kernel<<<gridSize(n), BLOCK_SIZE>>>(
        predictions.gpu(), labels.gpu(), grad.gpu(), n, inv_N);
    CUDA_CHECK(cudaGetLastError());
    return grad;
}
