#include "activations/ReLU.h"
#include "CudaUtils.h"

// Kernel para forward de ReLU en GPU
__global__ static void relu_forward_kernel(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = fmaxf(0.0f, data[i]);
}

// Kernel para backward de ReLU en GPU
__global__ static void relu_backward_kernel(float* grad, const float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= (output[i] > 0.0f) ? 1.0f : 0.0f;
}

void ReLU::forward(float* d_inout, int /*N*/, int total)
{
    relu_forward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_inout, total);
    CUDA_CHECK(cudaGetLastError());
}

void ReLU::backward(float* d_grad, const float* d_fwd_output, int /*N*/, int total)
{
    relu_backward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_grad, d_fwd_output, total);
    CUDA_CHECK(cudaGetLastError());
}
