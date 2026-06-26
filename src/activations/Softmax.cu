#include "activations/Softmax.h"
#include "CudaUtils.h"

// Kernel para forward de Softmax estable en GPU
__global__ static void softmax_forward_kernel(float* data, int N, int feature_size){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float* row = data + n * feature_size;

    float max_val = row[0];
    for (int i = 1; i < feature_size; ++i)
        max_val = fmaxf(max_val, row[i]);

    float sum = 0.0f;
    for (int i = 0; i < feature_size; ++i) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < feature_size; ++i)
        row[i] *= inv_sum;
}

// Kernel para backward de Softmax en GPU
__global__ static void softmax_backward_kernel(float* grad, const float* output, int N, int feature_size){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* out_row  = output + n * feature_size;
    float*       grad_row = grad   + n * feature_size;

    float dot = 0.0f;
    for (int i = 0; i < feature_size; ++i)
        dot += grad_row[i] * out_row[i];

    for (int i = 0; i < feature_size; ++i)
        grad_row[i] = out_row[i] * (grad_row[i] - dot);
}

void Softmax::forward(float* d_inout, int N, int total){
    int feature_size = total / N; 
    softmax_forward_kernel<<<gridSize(N), BLOCK_SIZE>>>(d_inout, N, feature_size);
    CUDA_CHECK(cudaGetLastError());
}

void Softmax::backward(float* d_grad, const float* d_fwd_output, int N, int total){
    int feature_size = total / N;
    softmax_backward_kernel<<<gridSize(N), BLOCK_SIZE>>>(d_grad, d_fwd_output, N, feature_size);
    CUDA_CHECK(cudaGetLastError());
}
