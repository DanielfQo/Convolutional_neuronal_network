#include "PoolLayer.h"
#include "CudaUtils.h"
#include <cstdio>
#include <stdexcept>

// Kernel para Max Pooling hacia adelante en GPU
__global__ void max_pool_forward_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float* __restrict__ mask,
        int N, int C, int inH, int inW,
        int outH, int outW,
        int pool, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow  = idx % outW;              idx /= outW;
    int oh  = idx % outH;              idx /= outH;
    int c   = idx % C;                 idx /= C;
    int n   = idx;

    int h_start = oh * stride;
    int w_start = ow * stride;

    float max_val = -1e38f;
    int   max_idx = -1;

    for (int kh = 0; kh < pool; ++kh) {
        for (int kw = 0; kw < pool; ++kw) {
            int ih = h_start + kh;
            int iw = w_start + kw;
            if (ih < inH && iw < inW) {
                int flat = ((n * C + c) * inH + ih) * inW + iw;
                if (input[flat] > max_val) {
                    max_val = input[flat];
                    max_idx = flat;
                }
            }
        }
    }

    int out_flat = ((n * C + c) * outH + oh) * outW + ow;
    output[out_flat] = max_val;
    mask[out_flat]   = static_cast<float>(max_idx);
}

// Kernel para Average Pooling hacia adelante en GPU
__global__ void avg_pool_forward_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int N, int C, int inH, int inW,
        int outH, int outW,
        int pool, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow  = idx % outW;              idx /= outW;
    int oh  = idx % outH;              idx /= outH;
    int c   = idx % C;                 idx /= C;
    int n   = idx;

    int h_start = oh * stride;
    int w_start = ow * stride;
    float sum = 0.f;
    int   cnt = 0;

    for (int kh = 0; kh < pool; ++kh) {
        for (int kw = 0; kw < pool; ++kw) {
            int ih = h_start + kh;
            int iw = w_start + kw;
            if (ih < inH && iw < inW) {
                sum += input[((n * C + c) * inH + ih) * inW + iw];
                ++cnt;
            }
        }
    }

    output[((n * C + c) * outH + oh) * outW + ow] = sum / static_cast<float>(cnt);
}

// Kernel para Max Pooling hacia atras
__global__ void max_pool_backward_kernel(
        const float* __restrict__ grad_out,
        const float* __restrict__ mask,
        float* __restrict__ grad_in,
        int total_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int src = static_cast<int>(mask[idx]);
    atomicAdd(&grad_in[src], grad_out[idx]);
}

// Kernel para Average Pooling hacia atras
__global__ void avg_pool_backward_kernel(
        const float* __restrict__ grad_out,
        float* __restrict__ grad_in,
        int N, int C, int inH, int inW,
        int outH, int outW,
        int pool, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * outH * outW;
    if (idx >= total_out) return;

    int ow  = idx % outW;              idx /= outW;
    int oh  = idx % outH;              idx /= outH;
    int c   = idx % C;                 idx /= C;
    int n   = idx;

    int h_start = oh * stride;
    int w_start = ow * stride;
    float g = grad_out[((n * C + c) * outH + oh) * outW + ow];

    int cnt = 0;
    for (int kh = 0; kh < pool; ++kh)
        for (int kw = 0; kw < pool; ++kw)
            if ((h_start + kh) < inH && (w_start + kw) < inW) ++cnt;

    float contrib = g / static_cast<float>(cnt);
    for (int kh = 0; kh < pool; ++kh) {
        for (int kw = 0; kw < pool; ++kw) {
            int ih = h_start + kh;
            int iw = w_start + kw;
            if (ih < inH && iw < inW)
                atomicAdd(&grad_in[((n * C + c) * inH + ih) * inW + iw], contrib);
        }
    }
}

// Constructor
PoolLayer::PoolLayer(int pool_size, int stride, PoolType pool_type)
    : pool_size_(pool_size),
      stride_(stride < 0 ? pool_size : stride),
      pool_type_(pool_type) {
    type_ = LayerType::POOL;
}

// Propagacion hacia adelante
Tensor PoolLayer::forward(const Tensor& input) {
    int N   = input.N();
    int C   = input.C();
    int inH = input.H();
    int inW = input.W();
    out_H_ = (inH - pool_size_) / stride_ + 1;
    out_W_ = (inW - pool_size_) / stride_ + 1;

    input_cache_ = Tensor(N, C, inH, inW);
    CUDA_CHECK(cudaMemcpy(input_cache_.gpu(), input.gpu(),
                          input.bytes(), cudaMemcpyDeviceToDevice));

    Tensor output(N, C, out_H_, out_W_);
    int total_out = output.size();

    if (pool_type_ == PoolType::MAX) {
        mask_ = std::make_unique<Tensor>(N, C, out_H_, out_W_);
        max_pool_forward_kernel<<<gridSize(total_out), BLOCK_SIZE>>>(
            input.gpu(), output.gpu(), mask_->gpu(),
            N, C, inH, inW, out_H_, out_W_, pool_size_, stride_);
    } else {
        avg_pool_forward_kernel<<<gridSize(total_out), BLOCK_SIZE>>>(
            input.gpu(), output.gpu(),
            N, C, inH, inW, out_H_, out_W_, pool_size_, stride_);
    }
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Propagacion hacia atras
Tensor PoolLayer::backward(const Tensor& grad_output) {
    int N   = input_cache_.N();
    int C   = input_cache_.C();
    int inH = input_cache_.H();
    int inW = input_cache_.W();

    Tensor grad_input(N, C, inH, inW);
    grad_input.zeros();

    if (pool_type_ == PoolType::MAX) {
        max_pool_backward_kernel<<<gridSize(grad_output.size()), BLOCK_SIZE>>>(
            grad_output.gpu(), mask_->gpu(), grad_input.gpu(), grad_output.size());
    } else {
        avg_pool_backward_kernel<<<gridSize(grad_output.size()), BLOCK_SIZE>>>(
            grad_output.gpu(), grad_input.gpu(),
            N, C, inH, inW, out_H_, out_W_, pool_size_, stride_);
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

// Guardar capa en archivo binario
void PoolLayer::save(std::ofstream& f) const {
    f.write(reinterpret_cast<const char*>(&pool_size_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&stride_),    sizeof(int));
    int pt = static_cast<int>(pool_type_);
    f.write(reinterpret_cast<const char*>(&pt),         sizeof(int));
}

// Cargar capa desde archivo binario
void PoolLayer::load(std::ifstream& f) {
    f.read(reinterpret_cast<char*>(&pool_size_), sizeof(int));
    f.read(reinterpret_cast<char*>(&stride_),    sizeof(int));
    int pt; f.read(reinterpret_cast<char*>(&pt), sizeof(int));
    pool_type_ = static_cast<PoolType>(pt);
}

// Resumen legible
std::string PoolLayer::summary() const {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "%sPool(size=%d, stride=%d)  act=none          params=0",
        pool_type_ == PoolType::MAX ? "Max" : "Avg",
        pool_size_, stride_);
    return std::string(buf);
}
