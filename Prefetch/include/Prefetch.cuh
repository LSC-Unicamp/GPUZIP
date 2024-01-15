#pragma once

#include "cudaUtils.cuh"
#include "structs.h"
#include <cuda_runtime.h>
#include <iostream>
#include <map>

#ifdef GPUZIP
#include "compressor.hpp"
#else
// If the current build has no compressor (GPUZIP), we have a dummy interface in
// order to keep Prefetch abstract class working.
template <typename decompressType, typename compressedType>
class compressor
{
public:
  size_t Compress(void *input, void *output)
  {
    fprintf(stderr,
            "Error: Compression support not enabled in this build. Please "
            "rebuild with compression support enabled (see project wiki).\n");
    exit(EXIT_FAILURE);
  }

  size_t Compressed_buffer_size(void *input)
  {
    fprintf(stderr,
            "Error: Compression support not enabled in this build. Please "
            "rebuild with compression support enabled (see project wiki).\n");
    exit(EXIT_FAILURE);
  }

  void Decompress(void *input, void *output)
  {
    fprintf(stderr,
            "Error: Compression support not enabled in this build. Please "
            "rebuild with compression support enabled (see project wiki).\n");
    exit(EXIT_FAILURE);
  }
};
#endif

/**
 * @file Prefetch.h
 * @brief The Prefetch class provides an abstract interface for prefetching
 *        operations with support for compression.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date November 25, 2023
 */
class Prefetch
{
protected:
  PrefetchAction_t prefetch_action;
  int iterator = 0;
  int bf_top = -1;
  ChkptPoolBf_t bf[2];
  unsigned bf_size = 2;
  ChkptPool_t pool;
  bool compression = false;
  unsigned snaps;
  unsigned timesteps;
  size_t max_len;
  Streams_t streams;
  const int info; ///< debug level 0,1,2 or 4.

  /**
   * @brief copyMetadata copies metadata information from the source to the
   *        destination field.
   *
   * @param dest The destination field.
   * @param src The source field.
   */
  virtual void copyMetadata(Field_t *dest, Field_t *src)
  {
    dest->size = src->size;
    dest->n1 = src->n1;
    dest->n2 = src->n2;
    dest->n3 = src->n3;
  }

  /**
   * @brief tobuff copies metadata and data from the source fields to the
   *        top of GPU buffer. Notice this is only GPU (computation data) to GPU
   * (buffer)
   *
   * @param curr The current field.
   * @param prev The previous field.
   */
  virtual void tobuff(Field_t *curr, Field_t *prev)
  {
    copyMetadata(bf[bf_top].curr, curr);
    PREFETCH_CUDA_CHECK(cudaMemcpyAsync(bf[bf_top].curr->data, curr->data,
                                        bf[bf_top].curr->size,
                                        cudaMemcpyDefault, streams.save));

    copyMetadata(bf[bf_top].prev, prev);
    PREFETCH_CUDA_CHECK(cudaMemcpyAsync(bf[bf_top].prev->data, prev->data,
                                        bf[bf_top].prev->size,
                                        cudaMemcpyDefault, streams.save));
  }

  /**
   * @brief buff2host copies metadata and data from the top of the buffer to
   *        the host pool.
   */
  virtual void buff2host()
  {
    copyMetadata(&(pool.currs[pool.top]), bf[bf_top].curr);
    PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
        pool.currs[pool.top].data, bf[bf_top].curr->data, bf[bf_top].curr->size,
        cudaMemcpyDefault, streams.save));

    copyMetadata(&(pool.prevs[pool.top]), bf[bf_top].prev);
    PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
        pool.prevs[pool.top].data, bf[bf_top].prev->data, bf[bf_top].prev->size,
        cudaMemcpyDefault, streams.save));
  }

  /**
   * @brief walk advances the pool and buffer indices for a given timestep.
   *
   * @param timestep The timestep.
   */
  virtual void walk(unsigned timestep)
  {
    ++pool.top;

    if (pool.top >= pool.size)
    {
      fprintf(stderr, "\nError: there's no slots available to save field!\n");
      exit(0);
    }

    pool.timestep[pool.top] = timestep;

    bf_top = (bf_top + 1) % 2;

    bf[bf_top].timestep = timestep;
  }

  /**
   * @brief alignPool ensures that the pool is correctly aligned with the
   *        specified timestep.
   *
   * @param timestep The timestep.
   */
  virtual void alignPool(int timestep)
  {
    unsigned i = pool.top;

    if (pool.timestep[i] < timestep)
    {
      fprintf(stderr, "\nError: no field with timestep %d is on the pool!\n",
              timestep);
      exit(0);
    }
    else if (pool.timestep[i] > timestep)
    {
      i = (--pool.top);

      if (pool.timestep[i] != timestep)
      {
        fprintf(stderr, "\nError: timestep %d in pool out of order!\n",
                timestep);
        exit(0);
      }

      if (bf[bf_top].timestep != timestep)
      {
        // update the top in local buffer. The matching field may have been
        // prefetched to the botton position. If it was not, it will be a miss.
        bf_top = (bf_top + 1) % 2;
      }
    }
  }

  /**
   * @brief getSlot retrieves the index in the pool for a given timestep.
   *
   * @param timestep The timestep.
   * @return The index in the pool.
   */
  unsigned getSlot(int timestep)
  {
    int i;
    for (i = 0; i < pool.size; i++)
    {
      if (pool.timestep[i] == timestep)
      {
        break;
      }
    }
    return i;
  }

public:
  /**
   * @brief Constructor for the Prefetch class.
   *
   * @param numSnaps The number of snapshots.
   * @param timesteps The number of timesteps.
   * @param max_len The maximum length allowed for a field -- n1 * n2 * n3 *
   * sizeof(float).
   */
  Prefetch(int numSnaps, unsigned timesteps, size_t max_len, int info, uint _bf_size = 2)
      : snaps(numSnaps), timesteps(timesteps), max_len(max_len), info(info)
  {

    cudaStreamCreate(&streams.save);
    cudaStreamCreate(&streams.retrieve);

    bf_size = _bf_size;
    pool.size = numSnaps;
    pool.top = -1;
    PREFETCH_CUDA_CHECK(
        cudaMallocHost(&(pool.currs), numSnaps * sizeof(Field_t)));
    PREFETCH_CUDA_CHECK(
        cudaMallocHost(&(pool.prevs), numSnaps * sizeof(Field_t)));
    PREFETCH_CUDA_CHECK(
        cudaMallocHost(&(pool.timestep), numSnaps * sizeof(int)));

    for (unsigned p = 0; p < pool.size; p++)
    {
      pool.timestep[p] = -1;
      PREFETCH_CUDA_CHECK(cudaMallocHost(&(pool.currs[p].data), max_len));
      PREFETCH_CUDA_CHECK(cudaMallocHost(&(pool.prevs[p].data), max_len));
    }

    for (unsigned p = 0; p < bf_size; p++)
    {
      PREFETCH_CUDA_CHECK(cudaMallocHost(&(bf[p].curr), sizeof(Field_t)));
      PREFETCH_CUDA_CHECK(cudaMallocHost(&(bf[p].prev), sizeof(Field_t)));
      PREFETCH_CUDA_CHECK(cudaMalloc(&(bf[p].curr->data), max_len));
      PREFETCH_CUDA_CHECK(cudaMalloc(&(bf[p].prev->data), max_len));
    }
  }

  /**
   * @brief setup - Abstract method. Initializes the prefetching parameters.
   */
  virtual void setup()
  {
    if (info > 3)
    {
      fprintf(stderr, "Cleaning up Prefetch class\n");
    }

    iterator = 0;
    pool.top = -1;
    bf_top = -1;

    bf[0].timestep = -1;
    bf[1].timestep = -1;

    for (int i = 0; i < pool.size; i++)
    {
      pool.timestep[i] = -1;
    }

    for (int i = 0; i < 2000; i++)
    {
      prefetch_action.timestep[i] = -1;
      prefetch_action.iter[i] = -1;
    }
  };

  /**
   * @brief save saves the current and previous fields in the buffer and later
   * pushes to the checkpoint pool in the host.
   *
   * @param timestep The timestep.
   * @param curr The current field.
   * @param prev The previous field.
   * @param compcurr The compressor for the current field.
   * @param compprev The compressor for the previous field.
   * @return True if successful, false otherwise.
   */
  virtual bool save(unsigned timestep, Field_t *curr, Field_t *prev,
            compressor<void, void> *compcurr,
            compressor<void, void> *compprev)
  {
    walk(timestep);

    curr->size = compcurr->Compress(curr->data, bf[bf_top].curr->data);
    copyMetadata(bf[bf_top].curr, curr);

    #if defined(PRINT_COMPRESSION_RATIO)
      size_t uncompressed_len = curr->n1 * curr->n2 * curr->n3 * sizeof(float);
      double ratio = (double) uncompressed_len / (double) curr->size;
      fprintf(stderr, "compression_ratio_gpuzip, %lu, %lu, %.3f\n", 
              uncompressed_len,
              curr->size,
              ratio);
    #endif

    prev->size = compprev->Compress(prev->data, bf[bf_top].prev->data);
    copyMetadata(bf[bf_top].prev, prev);

    #if defined(PRINT_COMPRESSION_RATIO)
      uncompressed_len = prev->n1 * prev->n2 * prev->n3 * sizeof(float);
      ratio = (double) uncompressed_len / (double) prev->size;
      fprintf(stderr, "compression_ratio_gpuzip, %lu, %lu, %.3f\n", 
              uncompressed_len,
              prev->size,
              ratio);
    #endif

    buff2host();

    return true;
  }

  /**
   * @brief save saves the current and previous fields in the buffer without
   *        compression.
   *
   * @param timestep The timestep.
   * @param curr The current field.
   * @param prev The previous field.
   * @return True if successful, false otherwise.
   */
  virtual bool save(int timestep, Field_t *curr, Field_t *prev)
  {
    walk(timestep);
    tobuff(curr, prev);
    buff2host();
    return true;
  }

  virtual bool retrieve(int timestep, Field_t *curr, Field_t *prev) {
    bool ret = retrieve(timestep, curr, prev, streams.retrieve);
    PREFETCH_CUDA_CHECK(cudaStreamSynchronize(streams.retrieve));
    return ret;
  }

  /**
   * @brief retrieve retrieves the current and previous fields for a given
   *        timestep.
   *
   * @param timestep The timestep.
   * @param curr The current field.
   * @param prev The previous field.
   * @return True if successful, false otherwise.
   */
  virtual bool retrieve(int timestep, Field_t *curr, Field_t *prev, cudaStream_t stream)
  {
    alignPool(timestep);

    // Check if the desired timestep is already in the local buffer
    if (bf[bf_top].timestep == timestep)
    {
      if (info >= 2)
      {
        fprintf(stderr, "===> Buffer Hit: timestep %d in local buffer\n",
                timestep);
      }

      copyMetadata(curr, bf[bf_top].curr);
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(curr->data, bf[bf_top].curr->data,
                                          bf[bf_top].curr->size,
                                          cudaMemcpyDefault, stream));

      copyMetadata(prev, bf[bf_top].prev);
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(prev->data, bf[bf_top].prev->data,
                                          bf[bf_top].prev->size,
                                          cudaMemcpyDefault, stream));
    }
    else
    {
      if (info >= 2)
      {
        fprintf(stderr,
                "\n===> Buffer Miss: timestep %d is not in local pool. Local "
                "buffer is %d(%d) | %d(%d)\n",
                timestep, bf_top, bf[bf_top].timestep, (bf_top + 1) % 2,
                bf[(bf_top + 1) % 2].timestep);
      }
      // It is not in local buffer, so bring it from Host Mem
      copyMetadata(curr, &(pool.currs[pool.top]));
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(curr->data, pool.currs[pool.top].data,
                                          pool.currs[pool.top].size,
                                          cudaMemcpyDefault, stream));

      copyMetadata(prev, &(pool.prevs[pool.top]));
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(prev->data, pool.prevs[pool.top].data,
                                          pool.prevs[pool.top].size,
                                          cudaMemcpyDefault, stream));

      // Updating the local buffer. Need to wait for the transfer above to
      // finish, but now we can use the check_pts stream so computation can
      // resume.
      PREFETCH_CUDA_CHECK(cudaStreamSynchronize(stream));

      bf[bf_top].timestep = timestep;
      copyMetadata(bf[bf_top].curr, curr);
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(bf[bf_top].curr->data, curr->data,
                                          curr->size, cudaMemcpyDefault,
                                          stream));

      copyMetadata(bf[bf_top].prev, prev);
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(bf[bf_top].prev->data, prev->data,
                                          prev->size, cudaMemcpyDefault,
                                          stream));
    }

    return true;
  }

  /**
   * @brief retrieve retrieves the current and previous fields for a given
   *        timestep with decompression.
   *
   * @param timestep The timestep.
   * @param curr The current field.
   * @param prev The previous field.
   * @param compcurr The compressor for the current field.
   * @param compprev The compressor for the previous field.
   * @return True if successful, false otherwise.
   */
  virtual bool retrieve(unsigned timestep, Field_t *curr, Field_t *prev,
                compressor<void, void> *compcurr,
                compressor<void, void> *compprev)
  {
    alignPool(timestep);

    // Check if the desired timestep is already in the local buffer
    if (bf[bf_top].timestep == timestep)
    {
      copyMetadata(curr, bf[bf_top].curr);
      curr->size = curr->n1 * curr->n2 * curr->n3 * sizeof(float);
      compcurr->Decompress(bf[bf_top].curr->data, curr->data);

      copyMetadata(prev, bf[bf_top].prev);
      prev->size = prev->n1 * prev->n2 * prev->n3 * sizeof(float);
      compprev->Decompress(bf[bf_top].prev->data, prev->data);
    }
    else
    {
      // It is not in local buffer, so bring it from Host Mem
      copyMetadata(bf[bf_top].curr, &(pool.currs[pool.top]));
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
          bf[bf_top].curr->data, pool.currs[pool.top].data,
          pool.currs[pool.top].size, cudaMemcpyDefault, streams.retrieve));

      copyMetadata(bf[bf_top].prev, &(pool.prevs[pool.top]));
      PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
          bf[bf_top].prev->data, pool.prevs[pool.top].data,
          pool.prevs[pool.top].size, cudaMemcpyDefault, streams.retrieve));

      PREFETCH_CUDA_CHECK(cudaStreamSynchronize(streams.retrieve));

      bf[bf_top].timestep = timestep;

      copyMetadata(curr, bf[bf_top].curr);
      curr->size = curr->n1 * curr->n2 * curr->n3 * sizeof(float);
      compcurr->Decompress(bf[bf_top].curr->data, curr->data);

      copyMetadata(prev, bf[bf_top].prev);
      prev->size = prev->n1 * prev->n2 * prev->n3 * sizeof(float);
      compprev->Decompress(bf[bf_top].prev->data, prev->data);
    }

    return true;
  }

  /**
   * @brief prefetch prefetches if necessary for a given iteration
   *
   * @param it The iteration.
   * @return True if prefetch was done, false otherwise.
   */
  virtual bool prefetch(int it)
  {
    if (prefetch_action.iter[iterator] != it)
    {
      return false;
    }

    unsigned pool_idx, timestep;

    timestep = prefetch_action.timestep[iterator];

    // Get index for the timestep in host pool. This points to fpcurr.
    pool_idx = getSlot(timestep);

    if (pool_idx == pool.size || pool.timestep[pool_idx] != timestep)
    {
      return false;
    }

    if (info >= 2)
    {
      fprintf(stderr, "\n===> Prefetching: pool_idx=%i , ts=%i, bf_top=%i\n", pool_idx, timestep, bf_top);
    }

    // Load the fields in the top of local buffer
    bf[bf_top].timestep = timestep;

    copyMetadata(bf[bf_top].curr, &(pool.currs[pool_idx]));

    PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
        bf[bf_top].curr->data, pool.currs[pool_idx].data,
        pool.currs[pool_idx].size, cudaMemcpyDefault, streams.save));

    copyMetadata(bf[bf_top].prev, &(pool.prevs[pool_idx]));

    PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
        bf[bf_top].prev->data, pool.prevs[pool_idx].data,
        pool.prevs[pool_idx].size, cudaMemcpyDefault, streams.save));

    iterator++;

    return true;
  }

  /**
   * @brief free frees allocated memory for streams, pool, and buffer.
   */
  void free()
  {
    cudaStreamDestroy(streams.save);
    cudaStreamDestroy(streams.retrieve);

    for (unsigned p = 0; p < pool.size; p++)
    {
      if (pool.currs[p].data != NULL)
      {
        PREFETCH_CUDA_CHECK(cudaFreeHost(pool.currs[p].data));
      }
      if (pool.prevs[p].data != NULL)
      {
        PREFETCH_CUDA_CHECK(cudaFreeHost(pool.prevs[p].data));
      }
    }

    if (pool.currs != NULL)
    {
      PREFETCH_CUDA_CHECK(cudaFreeHost(pool.currs));
    }

    if (pool.prevs != NULL)
    {
      PREFETCH_CUDA_CHECK(cudaFreeHost(pool.prevs));
    }

    if (pool.timestep != NULL)
    {
      PREFETCH_CUDA_CHECK(cudaFreeHost(pool.timestep));
    }

    for (unsigned p = 0; p < bf_size; p++)
    {
      if (bf[p].curr->data != NULL)
      {
        PREFETCH_CUDA_CHECK(cudaFree(bf[p].curr->data));
      }

      if (bf[p].prev->data != NULL)
      {
        PREFETCH_CUDA_CHECK(cudaFree(bf[p].prev->data));
      }

      if (bf[p].curr != NULL)
      {
        PREFETCH_CUDA_CHECK(cudaFreeHost(bf[p].curr));
      }

      if (bf[p].prev != NULL)
      {
        PREFETCH_CUDA_CHECK(cudaFreeHost(bf[p].prev));
      }
    }
  }

  /**
   * @brief log_buffer prints information about the GPU buffer.
   *
   * @param message A message to be printed before the table.
   */
  void log_buffer(const char *message)
  {
    fprintf(stderr, "GPU Buffer: %s\n", message);
    fprintf(stderr, "%-6s%-10s%-10s%-8s\n", "Index", "Timestep", "Size",
            "Dims");

    for (int i = 0; i < 2; ++i)
    {
      fprintf(stderr, "%-6d", i);
      fprintf(stderr, "%-10u", bf[i].timestep);
      fprintf(stderr, "%-10zu", bf[i].curr->size);
      fprintf(stderr, "%zux%zux%zu", bf[i].curr->n1, bf[i].curr->n2,
              bf[i].curr->n3);
      fprintf(stderr, "\n");

      fprintf(stderr, "%-6d", i);
      fprintf(stderr, "%-10u", bf[i].timestep);
      fprintf(stderr, "%-10zu", bf[i].prev->size);
      fprintf(stderr, "%zux%zux%zu", bf[i].prev->n1, bf[i].prev->n2,
              bf[i].prev->n3);
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "=====================================\n");
  }

  /**
   * @brief log_pool prints information about the checkpoint pool.
   *
   * @param message A message to be printed before the table.
   */
  void log_pool(const char *message)
  {
    fprintf(stderr, "Checkpoint Pool (host): %s\n", message);
    fprintf(stderr, "%-6s%-10s%-10s%-8s\n", "Index", "Timestep", "Size",
            "Dims");

    for (unsigned i = 0; i < pool.size; ++i)
    {
      fprintf(stderr, "%-6d", i);
      fprintf(stderr, "%-10d", pool.timestep[i]);
      fprintf(stderr, "%-10zu", pool.currs[i].size);
      fprintf(stderr, "%zux%zux%zu", pool.currs[i].n1, pool.currs[i].n2,
              pool.currs[i].n3);
      fprintf(stderr, "\n");

      fprintf(stderr, "%-6d", i);
      fprintf(stderr, "%-10d", pool.timestep[i]);
      fprintf(stderr, "%-10zu", pool.prevs[i].size);
      fprintf(stderr, "%zux%zux%zu", pool.prevs[i].n1, pool.prevs[i].n2,
              pool.prevs[i].n3);
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "=====================================\n");
  }

  /**
   * @brief log_prefetch prints information about the prefetch action vector.
   */
  void log_prefetch()
  {
    fprintf(stderr, "Prefetch Action Vector\n");
    fprintf(stderr, "%-5s%-5s\n", "it", "ts");

    for (unsigned i = 0; i < 2000; ++i)
    {
      if (prefetch_action.iter[i] == -1)
      {
        continue;
      }

      fprintf(stderr, "%-5d", prefetch_action.iter[i]);
      fprintf(stderr, "%-5d", prefetch_action.timestep[i]);
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "=====================================\n");
  }
};
