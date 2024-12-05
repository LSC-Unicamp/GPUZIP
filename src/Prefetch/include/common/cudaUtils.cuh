#pragma once

#include <cuda_runtime.h>
#include <iostream>


#define GPUZIP_CUDA_CHECK(call) \
do { \
    cudaError_t cudaStatus = (call); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(cudaStatus), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void NvtxPush(const char *title) {
    #ifdef USE_NVTX
        nvtxRangePush(title);
    #endif
}

void NvtxPop() {
    #ifdef USE_NVTX
        nvtxRangePop();
    #endif
}