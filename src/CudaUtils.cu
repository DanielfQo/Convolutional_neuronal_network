#include "CudaUtils.h"
#include <cstdio>

// Inicializa CUDA
void initCuda()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "[ERROR] No se encontraron dispositivos CUDA.\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(0));

    printDeviceInfo();

    printf("[CUDA] Inicializado.\n\n");
}

// Libera los recursos de CUDA y reinicia el dispositivo
void cleanupCuda()
{
    CUDA_CHECK(cudaDeviceReset());
    printf("[CUDA] Dispositivo reiniciado \n");
}

// Imprime informacion del hardware CUDA disponible
void printDeviceInfo()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("#######################################################\n");
    printf("              CUDA Device Information                 \n");
    printf("#######################################################\n");

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        double mem_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        printf("  [%d] %-28s  CC %d.%d  %.1f GB  \n",
               i, prop.name,
               prop.major, prop.minor,
               mem_gb);
    }

    printf("#######################################################\n");
}
