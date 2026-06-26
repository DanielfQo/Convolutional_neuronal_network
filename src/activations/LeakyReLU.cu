#include "activations/LeakyReLU.h"
#include "CudaUtils.h"

// Kernel para forward de LeakyReLU en GPU
__global__ static void leaky_relu_forward_kernel(float* data, float alpha, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = (data[i] > 0.0f) ? data[i] : alpha * data[i];
}

// Kernel para backward de LeakyReLU en GPU
__global__ static void leaky_relu_backward_kernel(float* grad,const float* output,float alpha, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= (output[i] > 0.0f) ? 1.0f : alpha;
}

void LeakyReLU::forward(float* d_inout, int /*N*/, int total){
    leaky_relu_forward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_inout, alpha_, total);
    CUDA_CHECK(cudaGetLastError());
}

void LeakyReLU::backward(float* d_grad, const float* d_fwd_output,int /*N*/, int total){
    leaky_relu_backward_kernel<<<gridSize(total), BLOCK_SIZE>>>(d_grad, d_fwd_output, alpha_, total);
    CUDA_CHECK(cudaGetLastError());
}
