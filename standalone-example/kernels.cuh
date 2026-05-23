#pragma once
#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Ricker wavelet source (peak frequency f0 Hz, evaluated at time t seconds)
// ---------------------------------------------------------------------------
__host__ __device__ inline float ricker(float t, float f0) {
    const float pi  = 3.14159265f;
    float tau = pi * f0 * (t - 1.0f / f0);
    return (1.0f - 2.0f * tau * tau) * expf(-tau * tau);
}

// ---------------------------------------------------------------------------
// 3-D acoustic wave equation (2nd-order FD):
//   next[i] = 2*curr[i] - prev[i] + coef * laplacian(curr)[i] + source
// coef = (v * dt / dx)^2
// Neumann (mirror) boundary conditions on all faces.
// ---------------------------------------------------------------------------
__global__ void wave_step_kernel(
    float* __restrict__       next,
    const float* __restrict__ curr,
    const float* __restrict__ prev,
    int n1, int n2, int n3,
    float coef,
    float source_val,
    int src_i, int src_j, int src_k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= n1 || j >= n2 || k >= n3) return;

    auto idx = [n2, n3](int ii, int jj, int kk) {
        return (ii * n2 + jj) * n3 + kk;
    };

    int im = (i > 0)      ? i - 1 : 0;
    int ip = (i < n1 - 1) ? i + 1 : n1 - 1;
    int jm = (j > 0)      ? j - 1 : 0;
    int jp = (j < n2 - 1) ? j + 1 : n2 - 1;
    int km = (k > 0)      ? k - 1 : 0;
    int kp = (k < n3 - 1) ? k + 1 : n3 - 1;

    float lap = curr[idx(ip,j,k)] + curr[idx(im,j,k)]
              + curr[idx(i,jp,k)] + curr[idx(i,jm,k)]
              + curr[idx(i,j,kp)] + curr[idx(i,j,km)]
              - 6.0f * curr[idx(i,j,k)];

    int c = idx(i, j, k);
    next[c] = 2.0f * curr[c] - prev[c] + coef * lap;

    if (i == src_i && j == src_j && k == src_k)
        next[c] += source_val;
}

// ---------------------------------------------------------------------------
// Imaging condition: gradient += fwd_curr * adj_curr  (cross-correlation)
// ---------------------------------------------------------------------------
__global__ void imaging_kernel(
    float* __restrict__       gradient,
    const float* __restrict__ fwd_curr,
    const float* __restrict__ adj_curr,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gradient[i] += fwd_curr[i] * adj_curr[i];
}

// ---------------------------------------------------------------------------
// Helper: 3-D launch grid for a given block size
// ---------------------------------------------------------------------------
inline dim3 make_grid(int n1, int n2, int n3, dim3 block) {
    return dim3((n1 + block.x - 1) / block.x,
                (n2 + block.y - 1) / block.y,
                (n3 + block.z - 1) / block.z);
}
