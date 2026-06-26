cmake_minimum_required(VERSION 3.20)
project(ConfigurableCNN LANGUAGES CXX CUDA)

# ──────────────────────────────────────────────────────────────
# Language standards
# ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────────────
# GPU architecture — adjust to match your GPU:
#   Ampere  (RTX 30xx)  → 86
#   Ada     (RTX 40xx)  → 89
#   Hopper  (H100)      → 90
#   Blackwell (RTX 50xx)→ 100
# ──────────────────────────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING
    "Semicolon-separated list of CUDA compute capability versions (e.g. \"86;89\")")

# ──────────────────────────────────────────────────────────────
# CUDA toolkit
# ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ──────────────────────────────────────────────────────────────
# Source collection
# ──────────────────────────────────────────────────────────────
file(GLOB ACTIVATION_SOURCES "src/activations/*.cu")
file(GLOB LOSS_SOURCES       "src/losses/*.cu")

set(CORE_SOURCES
    src/CudaUtils.cu
    src/Tensor.cu
    src/ConvLayer.cu
    src/PoolLayer.cu
    src/FCLayer.cu
    src/Network.cu
    ${ACTIVATION_SOURCES}
    ${LOSS_SOURCES}
)

# ──────────────────────────────────────────────────────────────
# Core Library
# ──────────────────────────────────────────────────────────────
add_library(cnn_core STATIC ${CORE_SOURCES})
target_include_directories(cnn_core PUBLIC include)
target_link_libraries(cnn_core PUBLIC
    CUDA::curand
)
target_compile_options(cnn_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# ──────────────────────────────────────────────────────────────
# Main executable
# ──────────────────────────────────────────────────────────────
add_executable(cnn_main src/main.cu)
target_link_libraries(cnn_main PRIVATE cnn_core)

# ──────────────────────────────────────────────────────────────
# BloodMNIST Test executable
# ──────────────────────────────────────────────────────────────
add_executable(bloodmnist_test tests/bloodMNIST.cpp)
target_link_libraries(bloodmnist_test PRIVATE cnn_core)

# ──────────────────────────────────────────────────────────────
# Build info
# ──────────────────────────────────────────────────────────────
message(STATUS "CUDA Toolkit version : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures   : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Activation sources   : ${ACTIVATION_SOURCES}")
message(STATUS "Loss sources         : ${LOSS_SOURCES}")
