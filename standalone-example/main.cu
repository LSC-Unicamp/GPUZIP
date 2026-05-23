/**
 * GPUZIP Standalone Example
 *
 * Demonstrates the full GPUZIP API (prefetching + compression) on a dummy
 * 3-D acoustic wave adjoint simulation:
 *
 *   Forward pass  : propagate a Ricker-wavelet source forward in time,
 *                   letting GPUZIP save checkpoints as directed by the
 *                   chosen checkpointing algorithm.
 *
 *   Backward pass : propagate an adjoint wavefield backward in time,
 *                   restoring forward checkpoints from GPUZIP cache/host
 *                   and accumulating the cross-correlation imaging condition.
 *
 * Configuration is read from config.cfg (or a path passed as argv[1]).
 * GPUZIP headers are included directly — no pre-compilation required.
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "common/GPUZIPBuilders.cpp"
#include "common/GPUZIPConfig.h"

#include "config_reader.hpp"
#include "kernels.cuh"

// ---------------------------------------------------------------------------
// Convenience macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Shift triple-buffer: prev ← curr ← next  (device-to-device copy)
// ---------------------------------------------------------------------------
static void shift_buffers(float* curr, float* prev, const float* next,
                           size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(prev, curr, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(curr, next, bytes, cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
// ---------------------------------------------------------------------------
{
    const std::string cfg_path = (argc > 1) ? argv[1] : "config.cfg";

    ConfigReader      cfg(cfg_path);
    gpuzip_config_t   gcfg = load_gpuzip_config(cfg);

    const int    n1    = cfg.get_int("n1",    64);
    const int    n2    = cfg.get_int("n2",    64);
    const int    n3    = cfg.get_int("n3",    64);
    const int    steps = cfg.get_int("steps", 300);
    const size_t N     = static_cast<size_t>(n1) * n2 * n3;
    const size_t bytes = N * sizeof(float);

    // Physics: velocity=1500 m/s, dt=1 ms, dx=10 m  →  CFL ≈ 0.15 (stable)
    const float v    = 1500.0f;
    const float dt   = 0.001f;
    const float dx   = 10.0f;
    const float coef = (v * dt / dx) * (v * dt / dx);
    const float f0   = 20.0f;   // Ricker peak frequency [Hz]

    // Source at grid centre; receiver at 3/4 of n1
    const int src_i = n1 / 2,  src_j = n2 / 2, src_k = n3 / 2;
    const int rec_i = 3*n1/4,  rec_j = n2 / 2, rec_k = n3 / 2;

    std::cout << "=== GPUZIP Standalone Example ===\n"
              << "Grid        : " << n1 << " x " << n2 << " x " << n3 << "\n"
              << "Steps       : " << steps << "\n"
              << "Checkpointing: " << (gcfg.checkpointing_algorithm ? "Revolve" : "Trace") << "\n"
              << "Cache capacity: " << gcfg.cache_capacity << "\n"
              << "Compressor  : " << gcfg.compressor
              << (gcfg.compressor == 0 ? " (none)" :
                  gcfg.compressor == 1 ? " (cuZFP)" :
                  gcfg.compressor == 2 ? " (Bitcomp)" : " (cuSZp)") << "\n"
              << std::flush;

    // -----------------------------------------------------------------------
    // GPU buffer allocation
    // -----------------------------------------------------------------------
    float *d_fwd_curr, *d_fwd_prev, *d_fwd_next;   // forward wavefield
    float *d_adj_curr, *d_adj_prev, *d_adj_next;   // adjoint wavefield
    float *d_gradient;                              // imaging result

    for (float** p : {&d_fwd_curr, &d_fwd_prev, &d_fwd_next,
                      &d_adj_curr, &d_adj_prev, &d_adj_next, &d_gradient}) {
        CUDA_CHECK(cudaMalloc(p, bytes));
        CUDA_CHECK(cudaMemset(*p, 0, bytes));
    }

    // Field_t descriptors — data pointers are fixed; GPUZIP reads/writes here
    Field_t fwd_curr_f = { d_fwd_curr, bytes, (size_t)n1, (size_t)n2, (size_t)n3 };
    Field_t fwd_prev_f = { d_fwd_prev, bytes, (size_t)n1, (size_t)n2, (size_t)n3 };

    // -----------------------------------------------------------------------
    // GPUZIP setup
    // -----------------------------------------------------------------------
    GPUZIPLogger::SetLevel(gcfg.log_level);

    Checkpointing* chkpt = GPUZIPBuilders::CheckpointingBuilder(&gcfg, steps);

    std::cout << "Snapshots   : " << chkpt->GetNumberOfCheckpoints() << "\n\n";

    Prefetch* prefetch = GPUZIPBuilders::PrefetchBuilder(
        &gcfg, (size_t)n1, (size_t)n2, (size_t)n3, steps, chkpt);

    // Two separate compressor instances (one per field) as per GPUZIP docs
    auto comp_curr = GPUZIPBuilders::CompressorBuilder(&gcfg, n1, n2, n3);
    auto comp_prev = GPUZIPBuilders::CompressorBuilder(&gcfg, n1, n2, n3);

    const bool use_compression = (gcfg.compressor > 0)
                                  && comp_curr && comp_prev;

    // PSA dry-run: builds the prefetch schedule (PAV) before the main loop
    std::cout << "Running PSA (prefetch setup algorithm)..." << std::flush;
    prefetch->Setup();
    std::cout << " done.\n\n";

    // -----------------------------------------------------------------------
    // Launch config for kernels
    // -----------------------------------------------------------------------
    const dim3 block(8, 8, 8);
    const dim3 grid = make_grid(n1, n2, n3, block);
    const int  flat_block = 256;
    const int  flat_grid  = static_cast<int>((N + flat_block - 1) / flat_block);

    // -----------------------------------------------------------------------
    // Main adjoint loop
    // -----------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();

    chkpt->Reset();
    Action action = chkpt->GetAction();

    while (action.actionType != ACTION_TERMINATE
           && action.actionType != ACTION_ERROR)
    {
        const int ts = action.ts;

        // Trigger prefetch transfers scheduled for this iteration
        prefetch->Dispatch(chkpt->GetIt());

        switch (action.actionType) {

        case ACTION_SAVE:
            // Sync before save so the kernel writing d_fwd_curr is complete
            CUDA_CHECK(cudaDeviceSynchronize());
            if (use_compression)
                prefetch->Save((unsigned)ts, &fwd_curr_f, &fwd_prev_f,
                               comp_curr.get(), comp_prev.get());
            else
                prefetch->Save(ts, &fwd_curr_f, &fwd_prev_f);
            break;

        case ACTION_RESTORE:
            if (use_compression)
                prefetch->Retrieve((unsigned)ts, &fwd_curr_f, &fwd_prev_f,
                                   comp_curr.get(), comp_prev.get());
            else
                prefetch->Retrieve(ts, &fwd_curr_f, &fwd_prev_f);
            // After restore, reset size fields to full uncompressed size
            fwd_curr_f.size = bytes;
            fwd_prev_f.size = bytes;
            break;

        case ACTION_FORWARD: {
            float src = ricker(ts * dt, f0);
            wave_step_kernel<<<grid, block>>>(
                d_fwd_next, d_fwd_curr, d_fwd_prev,
                n1, n2, n3, coef, src, src_i, src_j, src_k);
            shift_buffers(d_fwd_curr, d_fwd_prev, d_fwd_next, bytes);
            break;
        }

        case ACTION_BACKWARD: {
            // Imaging condition: accumulate cross-correlation
            imaging_kernel<<<flat_grid, flat_block>>>(
                d_gradient, d_fwd_curr, d_adj_curr, (int)N);

            // Advance adjoint wavefield (dummy receiver injection at rec_i)
            float adj_src = ricker(ts * dt, f0) * 1e-3f;
            wave_step_kernel<<<grid, block>>>(
                d_adj_next, d_adj_curr, d_adj_prev,
                n1, n2, n3, coef, adj_src, rec_i, rec_j, rec_k);
            shift_buffers(d_adj_curr, d_adj_prev, d_adj_next, bytes);
            break;
        }

        default:
            break;
        }

        action = chkpt->GetAction();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    if (action.actionType == ACTION_ERROR) {
        std::cerr << "ERROR: GPUZIP returned ACTION_ERROR. "
                     "Check log output above.\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // Results
    // -----------------------------------------------------------------------
    std::vector<float> h_gradient(N);
    CUDA_CHECK(cudaMemcpy(h_gradient.data(), d_gradient, bytes,
                          cudaMemcpyDeviceToHost));

    double grad_sum = 0.0;
    for (float v : h_gradient) grad_sum += v;

    std::cout << "Elapsed         : " << elapsed << " s\n"
              << "Gradient L1 sum : " << grad_sum
              << (std::abs(grad_sum) > 0 ? "  (non-zero — simulation ran)\n"
                                         : "  (zero — check config)\n");

    std::cout << "\n--- GPUZIP Cache Report ---\n";
    prefetch->Report();

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    prefetch->Free();
    delete prefetch;
    delete chkpt;

    for (float* p : {d_fwd_curr, d_fwd_prev, d_fwd_next,
                     d_adj_curr, d_adj_prev, d_adj_next, d_gradient})
        CUDA_CHECK(cudaFree(p));

    return 0;
}
