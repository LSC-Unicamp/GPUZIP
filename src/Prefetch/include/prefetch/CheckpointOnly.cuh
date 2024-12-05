#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#include "./Prefetch.cuh"

/**
 * @brief A subclass of Prefetch but without prefetching.
 *
 * This used to be how the application used to work before of GPUZIP
 * but we wrapped it into a class.
 *
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date November 25, 2023
 */
class CheckpointOnly : public Prefetch {

protected:
  void PrefetchSetupAlgorithm() override {
    // No prefetch here, right?
  }

  /**
   * @brief Saves the current and previous fields in the checkpoint pool (host
   * memory).
   *
   * Asynchronously copies data from GPU to host.
   * Synchronizes with the host to ensure data integrity.
   *
   * @param timestep The timestep associated with this checkpoint.
   * @param curr Pointer to the current field data.
   * @param prev Pointer to the previous field data.
   * @return True if successful, false otherwise.
   */
  bool save(int timestep, Field_t *curr, Field_t *prev) override {
    walk(timestep);

    copyMetadata(&(pool.currs[pool.top]), curr);
    GPUZIP_CUDA_CHECK(cudaMemcpyAsync(pool.currs[pool.top].data, curr->data,
                                      curr->size, cudaMemcpyDefault,
                                      streams.cache2host));

    copyMetadata(&(pool.prevs[pool.top]), prev);
    GPUZIP_CUDA_CHECK(cudaMemcpyAsync(pool.prevs[pool.top].data, prev->data,
                                      prev->size, cudaMemcpyDefault,
                                      streams.cache2host));

    // We need to wait for d2h copy otherwise
    // computation will override the allocated array.
    cudaStreamSynchronize(streams.cache2host);

    return true;
  }

  /**
   * @brief Restores the checkpoint data for the specified timestep.
   *
   * Asynchronously copies data from the checkpoint pool (host) back to the GPU.
   * Client is responsible for synchronization.
   *
   * @param timestep The timestep to restore.
   * @param curr Pointer to the current field data.
   * @param prev Pointer to the previous field data.
   * @param stream The CUDA stream for asynchronous operations.
   * @return True if successful, false otherwise.
   */
  bool retrieve(int timestep, Field_t *curr, Field_t *prev,
                cudaStream_t stream) override {
    alignPool(timestep);

    copyMetadata(curr, &(pool.currs[pool.top]));
    GPUZIP_CUDA_CHECK(cudaMemcpyAsync(curr->data, pool.currs[pool.top].data,
                                      pool.currs[pool.top].size,
                                      cudaMemcpyDefault, stream));

    copyMetadata(prev, &(pool.prevs[pool.top]));
    GPUZIP_CUDA_CHECK(cudaMemcpyAsync(prev->data, pool.prevs[pool.top].data,
                                      pool.prevs[pool.top].size,
                                      cudaMemcpyDefault, stream));
    return true;
  }

  /**
   * @brief Prefetch operation not implemented for this subclass.
   *
   * @param it The iteration number.
   * @return Always returns false as prefetching is not supported.
   */
  bool prefetch(int it) override { return false; }

public:
  /**
   * @brief Constructs a CheckpointOnly instance.
   *
   * Initializes a checkpointing mechanism without prefetching, using the given
   * parameters.
   *
   * @param numSnaps Number of snapshots to manage.
   * @param timesteps Total number of timesteps.
   * @param biggest_checkpoint_len Maximum data length for a field.
   * @param chkpt Pointer to the checkpointing mechanism.
   * @param compressionEnabled Specifies whether compression is enabled, which
   * determines buffer allocation for compressed data in GPU memory (default:
   * false).
   */
  CheckpointOnly(int numSnaps, int timesteps, size_t biggest_checkpoint_len,
                 Checkpointing *chkpt, bool compressionEnabled = false)
      : Prefetch(numSnaps, timesteps, biggest_checkpoint_len, chkpt,
                 getCacheSize(compressionEnabled)) {}

private:
  /**
   * @brief Retrieves the cache size for the checkpoint mechanism.
   *
   * @param compressionEnabled Indicates whether compression is enabled.
   * @return The cache size: returns 1 if compression is enabled, otherwise 0.
   */
  int getCacheSize(bool compressionEnabled) const {
    if (compressionEnabled) {
      return 1;
    } else {
      return 0;
    }
  }
};
