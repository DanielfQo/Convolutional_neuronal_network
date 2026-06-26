#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macros para comprobar errores en CUDA
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s  (file %s, line %d)\n",          \
                    cudaGetErrorString(_err), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Configurar tamano de bloque por defecto en CUDA
constexpr int BLOCK_SIZE = 256;

// Retorna el numero de bloques necesarios en base al tamano total
inline int gridSize(int n)
{
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Inicializa y limpia recursos de CUDA
void initCuda();
void cleanupCuda();
void printDeviceInfo();

