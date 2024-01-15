#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#include "../Prefetch.cuh"

/*
 * RevolvePrefetch: A subclass of Prefetch for Revolve-specific prefetching.
 * Author: Thiago Maltempi <tmaltempi@ic.unicamp.br>
 * Author: Sandro Rigo <srigo@ic.unicamp.br>
 * Date: November 25, 2023
 */
class CheckpointOnly : public Prefetch
{

protected:

    void *curr_compressed_data;
    void *prev_compressed_data;

    void walk(unsigned timestep)
    {
        ++pool.top;

        if (pool.top >= pool.size)
        {
            fprintf(stderr, "\nError: there's no slots available to save field!\n");
            exit(0);
        }

        pool.timestep[pool.top] = timestep;
    }

    void alignPool(int timestep)
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
        }
    }

public:
    CheckpointOnly(int numSnaps, int timesteps, size_t max_len, int info = 0)
        : Prefetch(numSnaps, timesteps, max_len, info, get_buffer_size()) {
    }

    /*
     * Setup method for RevolvePrefetch. Overrides the base class method.
     */
    void setup() override
    {
        if (info > 3)
        {
            fprintf(stderr, "Cleaning up CheckpointOnly class\n");
        }

        iterator = 0;
        pool.top = -1;

        for (int i = 0; i < pool.size; i++)
        {
            pool.timestep[i] = -1;
        }

        // Buffer top is always 0, since it is a single position only for compressed data.
        bf_top = 0;
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
    bool save(int timestep, Field_t *curr, Field_t *prev) override
    {
        walk(timestep);
        copyMetadata(&(pool.currs[pool.top]), curr);
        PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
            pool.currs[pool.top].data, curr->data, curr->size,
            cudaMemcpyDefault, streams.save));

        copyMetadata(&(pool.prevs[pool.top]), prev);
        PREFETCH_CUDA_CHECK(cudaMemcpyAsync(
            pool.prevs[pool.top].data, prev->data, prev->size,
            cudaMemcpyDefault, streams.save));
        return true;
    }

    bool retrieve(int timestep, Field_t *curr, Field_t *prev, cudaStream_t stream) override
    {
        alignPool(timestep);

        copyMetadata(curr, &(pool.currs[pool.top]));
        PREFETCH_CUDA_CHECK(cudaMemcpyAsync(curr->data, pool.currs[pool.top].data,
                                            pool.currs[pool.top].size,
                                            cudaMemcpyDefault, stream));

        copyMetadata(prev, &(pool.prevs[pool.top]));
        PREFETCH_CUDA_CHECK(cudaMemcpyAsync(prev->data, pool.prevs[pool.top].data,
                                            pool.prevs[pool.top].size,
                                            cudaMemcpyDefault, stream));

        return true;
    }

    bool prefetch(int it) override
    {
        return false;
    }

private:
    int get_buffer_size() const {
        #ifdef GPUZIP
            return 1;
        #else
            return 0;
        #endif
    }
};
