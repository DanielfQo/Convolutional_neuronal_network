#include "activations/Sigmoid.h"
#include "CudaUtils.h"

// Kernel para forward de Sigmoide en GPU
__global__ static void sigmoid_forward_kernel(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 1.0f / (1.0f + expf(-data[i]));
}

// Kernel para backward de Sigmoide en GPU
__global__ static void sigmoid_backward_kernel(float* grad, const float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= output[i] * (1.0f - output[i]);
}

void Sigmoid::forward(float* d_inout, int /*N*/, int total)
{
    sigmoid_forward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_inout, total);
    CUDA_CHECK(cudaGetLastError());
}

void Sigmoid::backward(float* d_grad, const float* d_fwd_output, int /*N*/, int total)
{
    sigmoid_backward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_grad, d_fwd_output, total);
    CUDA_CHECK(cudaGetLastError());
}
