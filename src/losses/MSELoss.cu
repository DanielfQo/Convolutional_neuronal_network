#include "losses/MSELoss.h"
#include "CudaUtils.h"

// Kernel para forward de MSE en GPU
__global__ void mse_loss_kernel(const float* pred, const float* label,
                                float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float d = pred[i] - label[i]; out[i] = d * d; }
}

// Kernel para backward de MSE en GPU
__global__ void mse_grad_kernel(const float* pred, const float* label,
                                float* grad, int n, float inv_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.f * (pred[i] - label[i]) * inv_N;
}

// Funcion auxiliar para sumar valores en CPU
static float gpuSumMSE(const float* d_buf, int n) {
    float* h = new float[n];
    cudaMemcpy(h, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += h[i];
    delete[] h;
    return s;
}

float MSELoss::compute(const Tensor& predictions, const Tensor& labels) {
    int n = predictions.size();
    float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));
    mse_loss_kernel<<<gridSize(n), BLOCK_SIZE>>>(predictions.gpu(), labels.gpu(), d_tmp, n);
    CUDA_CHECK(cudaGetLastError());
    float loss = gpuSumMSE(d_tmp, n) / static_cast<float>(predictions.N());
    CUDA_CHECK(cudaFree(d_tmp));
    return loss;
}

Tensor MSELoss::gradient(const Tensor& predictions, const Tensor& labels) {
    int n = predictions.size();
    Tensor grad(predictions.N(), predictions.C(), predictions.H(), predictions.W());
    float inv_N = 1.f / static_cast<float>(predictions.N());
    mse_grad_kernel<<<gridSize(n), BLOCK_SIZE>>>(
        predictions.gpu(), labels.gpu(), grad.gpu(), n, inv_N);
    CUDA_CHECK(cudaGetLastError());
    return grad;
}
