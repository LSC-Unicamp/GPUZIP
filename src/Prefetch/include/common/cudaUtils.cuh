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

/**
 * @brief Bytes to allocate for a checkpoint slot of `len` bytes.
 *
 * Rounds up to the 2 MiB granularity in which cudaMalloc maps device memory, so
 * the rounding, which occupies device memory anyway, becomes usable capacity
 * instead of an unchecked tail. Host pool slots use the same capacity, so
 * whatever fits in a cache slot also fits in a pool slot.
 */
inline size_t GPUZIPSlotCapacity(size_t len) {
    const size_t granularity = (size_t)2 << 20;
    return ((len + granularity - 1) / granularity) * granularity;
}

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