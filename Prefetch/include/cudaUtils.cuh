#pragma once

#include <cuda_runtime.h>
#include <iostream>

#if defined(ENABLE_CUDA_CHECK) || defined(DEBUG) || defined(_DEBUG)


#define PREFETCH_CUDA_CHECK(call) \
do { \
    cudaError_t cudaStatus = (call); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(cudaStatus), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#else

#define PREFETCH_CUDA_CHECK(cmd) cmd

#endif