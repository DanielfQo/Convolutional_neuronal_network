#include "ConvLayer.h"
#include "CudaUtils.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

// Kernel para transformar imagen a columnas (im2col) en GPU
__global__ void im2col_kernel(
        const float* __restrict__ input,
        float* __restrict__ col,
        int C, int H, int W,
        int kH, int kW,
        int stride, int padding,
        int outH, int outW) {

    int col_elements = C * kH * kW * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= col_elements) return;

    int ow  =  idx % outW;               idx /= outW;
    int oh  =  idx % outH;               idx /= outH;
    int kw  =  idx % kW;                 idx /= kW;
    int kh  =  idx % kH;                 idx /= kH;
    int c   =  idx;

    int ih = oh * stride - padding + kh;
    int iw = ow * stride - padding + kw;

    float val = 0.f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        val = input[c * H * W + ih * W + iw];

    int row = c * kH * kW + kh * kW + kw;
    int col_idx = oh * outW + ow;
    col[row * (outH * outW) + col_idx] = val;
}

// Kernel para transformar columnas a imagen (col2im) acumulando gradientes
__global__ void col2im_kernel(
        const float* __restrict__ col,
        float* __restrict__ grad_in,
        int C, int H, int W,
        int kH, int kW,
        int stride, int padding,
        int outH, int outW) {

    int col_elements = C * kH * kW * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= col_elements) return;

    int ow  =  idx % outW;               int tmp = idx / outW;
    int oh  =  tmp % outH;               tmp /= outH;
    int kw  =  tmp % kW;                 tmp /= kW;
    int kh  =  tmp % kH;                 tmp /= kH;
    int c   =  tmp;

    int ih = oh * stride - padding + kh;
    int iw = ow * stride - padding + kw;

    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        int row     = c * kH * kW + kh * kW + kw;
        int col_idx = oh * outW + ow;
        atomicAdd(&grad_in[c * H * W + ih * W + iw],
                  col[row * (outH * outW) + col_idx]);
    }
}

// Kernel para sumar sesgo (bias)
__global__ void add_bias_kernel(float* out, const float* bias,
                                int out_ch, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * spatial) return;
    int c = idx / spatial;
    out[idx] += bias[c];
}

// Kernel para calcular el gradiente del sesgo
__global__ void grad_bias_kernel(const float* grad_out, float* grad_b,
                                 int out_ch, int N, int spatial) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= out_ch) return;
    float sum = 0.f;
    for (int n = 0; n < N; ++n)
        for (int s = 0; s < spatial; ++s)
            sum += grad_out[(n * out_ch + c) * spatial + s];
    grad_b[c] = sum;
}

// Kernels personalizados de multiplicacion de matrices para ConvLayer

__global__ void matmul_forward_conv_kernel(
    const float* __restrict__ col_ptr,      // [col_rows, spatial]
    const float* __restrict__ weights,      // [out_ch, col_rows]
    float* __restrict__ out_ptr,            // [out_ch, spatial]
    int out_ch, int spatial, int col_rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_ch * spatial;
    if (idx >= total) return;

    int c = idx / spatial;
    int s = idx % spatial;

    float sum = 0.0f;
    for (int k = 0; k < col_rows; ++k) {
        sum += weights[c * col_rows + k] * col_ptr[k * spatial + s];
    }
    out_ptr[idx] = sum;
}

__global__ void matmul_backward_w_conv_kernel(
    const float* __restrict__ col_ptr,      // [col_rows, spatial]
    const float* __restrict__ grad_out_ptr, // [out_ch, spatial]
    float* __restrict__ grad_w,             // [out_ch, col_rows]
    int out_ch, int col_rows, int spatial,
    float alpha, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_ch * col_rows;
    if (idx >= total) return;

    int c = idx / col_rows;
    int k = idx % col_rows;

    float sum = 0.0f;
    for (int s = 0; s < spatial; ++s) {
        sum += grad_out_ptr[c * spatial + s] * col_ptr[k * spatial + s];
    }
    grad_w[idx] = alpha * sum + beta * grad_w[idx];
}

__global__ void matmul_backward_col_conv_kernel(
    const float* __restrict__ grad_out_ptr, // [out_ch, spatial]
    const float* __restrict__ weights,      // [out_ch, col_rows]
    float* __restrict__ col_ptr,            // [col_rows, spatial]
    int out_ch, int col_rows, int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = col_rows * spatial;
    if (idx >= total) return;

    int k = idx / spatial;
    int s = idx % spatial;

    float sum = 0.0f;
    for (int c = 0; c < out_ch; ++c) {
        sum += weights[c * col_rows + k] * grad_out_ptr[c * spatial + s];
    }
    col_ptr[idx] = sum;
}

// Constructor
ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size,
                     int stride, int padding,
                     std::unique_ptr<IActivation> act)
    : in_ch_(in_channels), out_ch_(out_channels),
      ksize_(kernel_size), stride_(stride), padding_(padding),
      weights_(out_channels, in_channels, kernel_size, kernel_size),
      bias_(1, out_channels, 1, 1),
      grad_w_(out_channels, in_channels, kernel_size, kernel_size),
      grad_b_(1, out_channels, 1, 1) {

    type_ = LayerType::CONV;
    activation_ = std::move(act);

    float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
    weights_.randomNormal(0.f, std::sqrt(2.f / fan_in));
    bias_.zeros();
}

// Propagacion hacia adelante
Tensor ConvLayer::forward(const Tensor& input) {
    int N  = input.N();
    int H  = input.H();
    int W  = input.W();
    out_H_ = (H + 2 * padding_ - ksize_) / stride_ + 1;
    out_W_ = (W + 2 * padding_ - ksize_) / stride_ + 1;

    input_cache_ = Tensor(N, in_ch_, H, W);
    CUDA_CHECK(cudaMemcpy(input_cache_.gpu(), input.gpu(),
                          input.bytes(), cudaMemcpyDeviceToDevice));

    int col_rows = in_ch_ * ksize_ * ksize_;
    int col_cols = out_H_ * out_W_;
    col_cache_ = Tensor(1, 1, col_rows, col_cols);

    Tensor output(N, out_ch_, out_H_, out_W_);
    int spatial = out_H_ * out_W_;

    for (int n = 0; n < N; ++n) {
        const float* in_ptr = input.gpu() + n * in_ch_ * H * W;
        float* col_ptr = col_cache_.gpu();

        int col_elems = col_rows * col_cols;
        im2col_kernel<<<gridSize(col_elems), BLOCK_SIZE>>>(
            in_ptr, col_ptr,
            in_ch_, H, W,
            ksize_, ksize_,
            stride_, padding_,
            out_H_, out_W_);
        CUDA_CHECK(cudaGetLastError());

        float* out_ptr = output.gpu() + n * out_ch_ * spatial;
        matmul_forward_conv_kernel<<<gridSize(out_ch_ * spatial), BLOCK_SIZE>>>(
            col_ptr, weights_.gpu(), out_ptr, out_ch_, spatial, col_rows
        );
        CUDA_CHECK(cudaGetLastError());
    }

    add_bias_kernel<<<gridSize(N * out_ch_ * spatial), BLOCK_SIZE>>>(
        output.gpu(), bias_.gpu(), out_ch_, spatial);
    CUDA_CHECK(cudaGetLastError());

    if (activation_) {
        output_cache_ = output.clone();
        activation_->forward(output.gpu(), N, output.size());
    }

    return output;
}

// Propagacion hacia atras
Tensor ConvLayer::backward(const Tensor& grad_output) {
    int N       = input_cache_.N();
    int H       = input_cache_.H();
    int W       = input_cache_.W();
    int spatial = out_H_ * out_W_;

    Tensor grad = Tensor(N, out_ch_, out_H_, out_W_);
    CUDA_CHECK(cudaMemcpy(grad.gpu(), grad_output.gpu(),
                          grad_output.bytes(), cudaMemcpyDeviceToDevice));

    if (activation_) {
        activation_->backward(grad.gpu(), output_cache_.gpu(), N, grad.size());
    }

    int col_rows = in_ch_ * ksize_ * ksize_;
    int col_cols = out_H_ * out_W_;
    Tensor col_buf(1, 1, col_rows, col_cols);

    grad_w_.zeros();
    grad_b_.zeros();
    Tensor grad_input(N, in_ch_, H, W);
    grad_input.zeros();

    for (int n = 0; n < N; ++n) {
        const float* in_ptr  = input_cache_.gpu() + n * in_ch_ * H * W;
        float*       col_ptr = col_buf.gpu();

        int col_elems = col_rows * col_cols;
        im2col_kernel<<<gridSize(col_elems), BLOCK_SIZE>>>(
            in_ptr, col_ptr,
            in_ch_, H, W,
            ksize_, ksize_,
            stride_, padding_,
            out_H_, out_W_);
        CUDA_CHECK(cudaGetLastError());

        float* grad_out_ptr = grad.gpu() + n * out_ch_ * spatial;
        float alpha = 1.f, beta = 1.f;

        matmul_backward_w_conv_kernel<<<gridSize(out_ch_ * col_rows), BLOCK_SIZE>>>(
            col_ptr, grad_out_ptr, grad_w_.gpu(), out_ch_, col_rows, spatial, alpha, beta
        );
        CUDA_CHECK(cudaGetLastError());

        matmul_backward_col_conv_kernel<<<gridSize(col_rows * spatial), BLOCK_SIZE>>>(
            grad_out_ptr, weights_.gpu(), col_ptr, out_ch_, col_rows, spatial
        );
        CUDA_CHECK(cudaGetLastError());

        float* gin_ptr = grad_input.gpu() + n * in_ch_ * H * W;
        col2im_kernel<<<gridSize(col_elems), BLOCK_SIZE>>>(
            col_ptr, gin_ptr,
            in_ch_, H, W,
            ksize_, ksize_,
            stride_, padding_,
            out_H_, out_W_);
        CUDA_CHECK(cudaGetLastError());
    }

    grad_bias_kernel<<<gridSize(out_ch_), BLOCK_SIZE>>>(
        grad.gpu(), grad_b_.gpu(), out_ch_, N, spatial);
    CUDA_CHECK(cudaGetLastError());

    return grad_input;
}

// Kernel para actualizar pesos con SGD
__global__ void sgd_update_kernel(float* w, const float* gw, int n, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * gw[i];
}

// Actualizar pesos y sesgos
void ConvLayer::updateWeights(float lr) {
    sgd_update_kernel<<<gridSize(weights_.size()), BLOCK_SIZE>>>(
        weights_.gpu(), grad_w_.gpu(), weights_.size(), lr);
    sgd_update_kernel<<<gridSize(bias_.size()), BLOCK_SIZE>>>(
        bias_.gpu(), grad_b_.gpu(), bias_.size(), lr);
    CUDA_CHECK(cudaGetLastError());
}

// Guardar capa en archivo binario
void ConvLayer::save(std::ofstream& f) const {
    f.write(reinterpret_cast<const char*>(&in_ch_),  sizeof(int));
    f.write(reinterpret_cast<const char*>(&out_ch_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&ksize_),  sizeof(int));
    f.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    f.write(reinterpret_cast<const char*>(&padding_), sizeof(int));
    weights_.save(f);
    bias_.save(f);
}

// Cargar capa desde archivo binario
void ConvLayer::load(std::ifstream& f) {
    f.read(reinterpret_cast<char*>(&in_ch_),  sizeof(int));
    f.read(reinterpret_cast<char*>(&out_ch_), sizeof(int));
    f.read(reinterpret_cast<char*>(&ksize_),  sizeof(int));
    f.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    f.read(reinterpret_cast<char*>(&padding_), sizeof(int));
    weights_.load(f);
    bias_.load(f);
}

// Resumen legible
std::string ConvLayer::summary() const {
    char buf[256];
    int params = out_ch_ * in_ch_ * ksize_ * ksize_ + out_ch_;
    std::snprintf(buf, sizeof(buf),
        "Conv(%d->%d, k=%d, s=%d, p=%d)  act=%-10s  params=%d",
        in_ch_, out_ch_, ksize_, stride_, padding_,
        activation_ ? activation_->name().c_str() : "none",
        params);
    return std::string(buf);
}
