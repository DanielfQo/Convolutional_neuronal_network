#include "activations/Tanh.h"
#include "CudaUtils.h"

// Kernel para forward de Tanh en GPU
__global__ static void tanh_forward_kernel(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = tanhf(data[i]);
}

// Kernel para backward de Tanh en GPU
__global__ static void tanh_backward_kernel(float* grad, const float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= 1.0f - output[i] * output[i];
}

void Tanh::forward(float* d_inout, int /*N*/, int total)
{
    tanh_forward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_inout, total);
    CUDA_CHECK(cudaGetLastError());
}

void Tanh::backward(float* d_grad, const float* d_fwd_output, int /*N*/, int total)
{
    tanh_backward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_grad, d_fwd_output, total);
    CUDA_CHECK(cudaGetLastError());
}
