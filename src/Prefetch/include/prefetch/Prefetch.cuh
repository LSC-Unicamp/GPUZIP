#pragma once

#include "../cache/GPUCheckpointCache.cuh"
#include "../cache/GPUCheckpointCacheMemento.cuh"
#include "../common/GPUZIPLogger.cpp"
#include "../common/cudaUtils.cuh"
#include "../common/structs.h"

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <map>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "Compressor.hpp"

#include "checkpointing/Checkpointing.hpp"

/**
 * @file Prefetch.h
 * @brief The Prefetch class provides an abstract interface for prefetching
 *        operations with support for compression.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date November 25, 2023
 */
class Prefetch {
protected:
  PrefetchAction_t pav;
  int iterator = 0;
  GPUCheckpointCache *cache;
  int cache_capacity;
  ChkptPool_t pool;
  bool compression = false;
  unsigned snaps;
  unsigned timesteps;
  size_t biggest_checkpoint_len;
  Streams_t streams;
  Checkpointing *chkpt;
  int actualMissesCount = 0;
  int hitsCount = 0;
  int prefetchCount = 0;

  /**
   * @brief Sets up the prefetching algorithm.
   *
   * This method is responsible for configuring the necessary data structures
   * and resources needed for the prefetching mechanism.
   */
  virtual void PrefetchSetupAlgorithm() {
    auto start = std::chrono::system_clock::now();

    GPUCheckpointCache *dummycache = new GPUCheckpointCache(cache_capacity);

    chkpt->Reset();

    // article reffers memento as "State Recorder"
    GPUCheckpointCacheMemento *memento = new GPUCheckpointCacheMemento();

    int last_hit_it = -1;

    std::map<int, int> touches;
    std::map<int, int> actions;
    std::map<int, int> misses;

    Field_t f; // dummy field

    bool terminate = false;

    // Fill up the action vector for prefetching computation
    do {
      Action action = chkpt->GetAction();

      if (action.actionType == ACTION_SAVE) {
        GPUZIPLogger::Debug("%04i, %04i - TAKSHOT - ", chkpt->GetIt(),
                            action.ts);
        dummycache->push(action.ts, &f, &f, chkpt->GetIt());
        touches[chkpt->GetIt()] = action.ts;
        if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
          dummycache->print();
        }
        actions[chkpt->GetIt()] = 0;
      }

      if (action.actionType == ACTION_FORWARD) {
        GPUZIPLogger::Debug("%04i, %04i - FWD - ", chkpt->GetIt(), action.ts);
        if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
          dummycache->print();
        }
      }

      if (action.actionType == ACTION_BACKWARD) {
        GPUZIPLogger::Debug("%04i, %04i - BWD - ", chkpt->GetIt(), action.ts);
        if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
          dummycache->print();
        }
      }

      touches[chkpt->GetIt()] = action.ts;

      if (action.actionType == ACTION_RESTORE) {
        actions[chkpt->GetIt()] = 1;
        int hit_index = dummycache->getIndex(action.ts, chkpt->GetIt());

        if (hit_index > -1) {
          last_hit_it = chkpt->GetIt();
          GPUZIPLogger::Debug("%04i, %04i - RET_HIT - ", chkpt->GetIt(),
                              action.ts);
          if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
            dummycache->print();
          }
        } else { // miss
          bool has_space_for_prefetch = true;
          bool is_pav_inited =
              !pav.iter.empty() && pav.iter.size() >= cache_capacity;
          int proposed_prefetch_it = last_hit_it + 1;
          int proposed_prefetch_dummycachepos = 0;

          for (int i = dummycache->getLength(); i > 0; i--) {
            int last_used_it = dummycache->lastUsedAt(i - 1);
            if (last_hit_it > last_used_it) {
              if (!is_pav_inited) {
                proposed_prefetch_it = last_used_it + 1;
              } else {
                if (last_used_it + 1 > pav.iter.back()) {
                  proposed_prefetch_it = last_used_it + 1;
                }
              }
            }
          }

          int counter = 0;

          if (is_pav_inited) {
            // Iterates looking for other prefetches that will
            // happen in the propose iteration
            for (int i = 0; i < pav.iter.size(); i++) {
              if (pav.iter.at(i) == proposed_prefetch_it) {
                counter++;
              }
            }

            has_space_for_prefetch = counter == 0;
          }

          if (has_space_for_prefetch) {
            // Sanity check
            if (proposed_prefetch_dummycachepos == -1) {
              GPUZIPLogger::Error("PSA Error: Index is -1!\n");
              exit(0);
            }

            GPUZIPLogger::Debug("%04i, %04i - RET_PRE - ", chkpt->GetIt(),
                                action.ts);
            if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
              dummycache->print();
            }

            last_hit_it = chkpt->GetIt();
            pav.iter.push_back(proposed_prefetch_it);
            pav.timestep.push_back(action.ts);
            pav.pos.push_back(proposed_prefetch_dummycachepos);
            pav.whenused.push_back(chkpt->GetIt());

            GPUZIPLogger::Debug(
                "===> Predicted Cache Miss at: %d. Setting prefetch at "
                "%d for "
                "timestep %d. dummycachepos=%i\n",
                chkpt->GetIt(), proposed_prefetch_it, action.ts,
                proposed_prefetch_dummycachepos);

            for (int i = 0; i < dummycache->getCapacity(); i++) {
              dummycache->push(-1, &f, &f, -1);
            }

            memento->restore(proposed_prefetch_it - 1, dummycache);

            for (int j = proposed_prefetch_it - 1; j < chkpt->GetIt(); j++) {
              for (int k = 0; k < pav.timestep.size(); k++) {
                if (pav.iter[k] == j) {

                  dummycache->insertAt(pav.timestep[k], &f, &f, pav.pos[k],
                                       true, j);

                  memento->save(j, dummycache);
                }
              }

              if (actions.find(j) != actions.end()) {
                int ts = touches.find(j) != touches.end() ? touches[j] : -1;
                if (actions[j] == 0) { // save
                  dummycache->push(ts, &f, &f, j);
                } else { // restore
                  bool notMiss = misses.find(j) != misses.end();
                  if (dummycache->getIndex(ts, j) == -1 && notMiss) {
                    GPUZIPLogger::Error("PSA Error!!! %i - %i\n", ts, j);
                    dummycache->print();
                    exit(1);
                  }
                }
                memento->save(j, dummycache);
              }
            }

            if (dummycache->getIndex(action.ts, chkpt->GetIt()) == -1) {
              GPUZIPLogger::Error("PSA Error: It supposed to be a hit since we "
                                  "already prefetched it.\n");
              exit(1);
            }
          } else {
            dummycache->push(action.ts, &f, &f, chkpt->GetIt());
            GPUZIPLogger::Debug("%04i, %04i - RET_MIS - UNAVOIDABLE\n",
                                chkpt->GetIt(), action.ts);
            misses[chkpt->GetIt()] = 1;
          }
        }
      }

      memento->save(chkpt->GetIt(), dummycache);

      if (action.actionType == ACTION_TERMINATE) {
        GPUZIPLogger::Debug("<TERMINATE>\n");
        GPUZIPLogger::Info("Total iterations: %d\n", chkpt->GetIt());
        GPUZIPLogger::Info("[Prefetch] Total prefetches defined: %d\n",
                           pav.iter.size());
        terminate = true;
      }

      if (action.actionType == ACTION_ERROR) {
        terminate = true;
      }
    } while (!terminate);

    if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG)
      GPUZIPLogger::LogPAV(&pav);

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now() - start)
                        .count();

    GPUZIPLogger::Debug("Prefetch Setup executed in %llds\n", duration);
  }

  /**
   * @brief copyMetadata copies metadata information from the source to the
   *        destination field.
   *
   * @param dest The destination field.
   * @param src The source field.
   */
  virtual void copyMetadata(Field_t *dest, Field_t *src) {
    dest->size = src->size;
    dest->n1 = src->n1;
    dest->n2 = src->n2;
    dest->n3 = src->n3;
  }

  /**
   * @brief cache2host copies metadata and data from the top of the cache to
   *        the host pool.
   */
  virtual void cache2host(cudaStream_t stream) {
    copyMetadata(&(pool.currs[pool.top]), cache->getCurrTop());
    GPUZIP_CUDA_CHECK(
        cudaMemcpyAsync(pool.currs[pool.top].data, cache->getCurrTop()->data,
                        cache->getCurrTop()->size, cudaMemcpyDefault, stream));

    copyMetadata(&(pool.prevs[pool.top]), cache->getPrevTop());
    GPUZIP_CUDA_CHECK(
        cudaMemcpyAsync(pool.prevs[pool.top].data, cache->getPrevTop()->data,
                        cache->getPrevTop()->size, cudaMemcpyDefault, stream));
  }

  /**
   * @brief walk advances the pool and cache indices for a given timestep.
   *
   * @param timestep The timestep.
   */
  virtual void walk(unsigned timestep) {
    ++pool.top;

    if (pool.top >= pool.size) {
      GPUZIPLogger::Error(
          "\nError: there's no slots available to save field!\n");
      exit(1);
    }

    pool.timestep[pool.top] = timestep;
  }

  /**
   * @brief alignPool ensures that the pool is correctly aligned with the
   *        specified timestep.
   *
   * @param timestep The timestep.
   */
  virtual void alignPool(int timestep) {
    unsigned i = pool.top;

    if (pool.timestep[i] < timestep) {
      GPUZIPLogger::Error(
          "\nError: no field with timestep %d is on the pool!\n", timestep);
      exit(1);
    } else if (pool.timestep[i] > timestep) {
      i = (--pool.top);

      if (pool.timestep[i] != timestep) {
        GPUZIPLogger::Error("\nError: timestep %d in pool out of order!\n",
                            timestep);
        exit(1);
      }
    }
  }

  /**
   * @brief getSlot retrieves the index in the pool for a given timestep.
   *
   * @param timestep The timestep.
   * @return The index in the pool.
   */
  unsigned getSlot(int timestep) {
    int i;
    for (i = 0; i < pool.size; i++) {
      if (pool.timestep[i] == timestep) {
        break;
      }
    }
    return i;
  }

  /**
   * @brief Resets the prefetching state and clears all associated data
   * structures.
   *
   * This method reinitializes the prefetching system by resetting the
   * checkpointing mechanism, clearing the checkpoint pool, and resetting
   * relevant internal structures.
   *
   * @note This function should be called before starting a new prefetching
   * process to ensure a clean state.
   */
  void reset() {
    chkpt->Reset();

    iterator = 0;
    pool.top = -1;

    for (int i = 0; i < pool.size; i++) {
      pool.timestep[i] = -1;
    }

    pav.timestep.clear();
    pav.iter.clear();

    cache->clear();
  }

  /**
   * @brief save saves the current and previous fields in the cache and later
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
                    Compressor<void, void> *compcurr,
                    Compressor<void, void> *compprev) {

    walk(timestep);
    cache->push(timestep, curr, prev, 0);

    cache->getCurrTop()->size =
        compcurr->Compress(curr->data, cache->getCurrTop()->data);

    cache->getPrevTop()->size =
        compcurr->Compress(prev->data, cache->getPrevTop()->data);

    curr->size = cache->getCurrTop()->size;
    prev->size = cache->getPrevTop()->size;

    cache2host(cache->getPciTop());

    GPUZIPLogger::LogCompressionMetrics(curr, timestep, 0);
    GPUZIPLogger::LogCompressionMetrics(prev, timestep, 1);

    GPUZIPLogger::Debug("After Save: ts=%i ", timestep);
    if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
      cache->print();
    }

    return true;
  }

  /**
   * @brief save saves the current and previous fields in the cache without
   *        compression.
   *
   * @param timestep The timestep.
   * @param curr The current field.
   * @param prev The previous field.
   * @return True if successful, false otherwise.
   */
  virtual bool save(int timestep, Field_t *curr, Field_t *prev) {
    GPUZIPLogger::Debug("Before Save: ts=%i ", timestep);
    if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
      cache->print();
    }

    walk(timestep);

    cache->push(timestep, curr, prev, 0);

    if (timestep != cache->getTimestepTop()) {
      GPUZIPLogger::Error("I got timestep = %i and found %i on top\n", timestep,
                          cache->getTimestepTop());
      exit(1);
    }

    cudaStream_t d2d = cache->getD2DTop();

    GPUZIP_CUDA_CHECK(cudaMemcpyAsync(cache->getCurrTop()->data, curr->data,
                                      curr->size, cudaMemcpyDefault, d2d));

    GPUZIP_CUDA_CHECK(cudaMemcpyAsync(cache->getPrevTop()->data, prev->data,
                                      prev->size, cudaMemcpyDefault, d2d));

    cudaStreamSynchronize(d2d);

    cudaStream_t pci = cache->getPciTop();
    cache2host(pci);

    GPUZIPLogger::Debug("After Save: ts=%i ", timestep);
    if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
      cache->print();
    }

    return true;
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
  virtual bool retrieve(int timestep, Field_t *curr, Field_t *prev,
                        cudaStream_t stream) {
    alignPool(timestep);
    int hit_index = cache->findIndex(timestep);

    if (hit_index > -1) {

      NvtxPush("CACHE_HIT");

      hitsCount++;

      GPUZIPLogger::Debug(
          "[RETRIEVE_HIT] Cache Hit: timestep %d in local cache\n", timestep);
      GPUZIPLogger::Debug("Before Hit: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      if (cache->timestepAt(hit_index) != timestep) {
        GPUZIPLogger::Error(
            "[RETRIEVE_HIT] I got timestep = %i and found %i at position %i\n",
            timestep, cache->timestepAt(hit_index), hit_index);
        exit(1);
      }

      cudaStream_t pci = cache->getPciAt(hit_index);
      cudaStream_t d2d = cache->getD2DAt(hit_index);

      if (cache->isPrefetched(hit_index)) {
        // wait for prefetching action get completed if it still transfering...
        prefetchCount++;

        NvtxPush("WAIT_PCI");

        cudaStreamSynchronize(pci);
#ifdef USE_NVTX
        nvtxRangePop();
#endif
      } else {
        GPUZIPLogger::Debug(
            "[RETRIEVE_HIT] timestep %i was NOT prefetched. There's "
            "nothing to "
            "wait for.\n",
            timestep);
      }

      copyMetadata(curr, cache->getCurrFieldAt(hit_index));
      GPUZIP_CUDA_CHECK(cudaMemcpyAsync(
          curr->data, cache->getCurrFieldAt(hit_index)->data,
          cache->getCurrFieldAt(hit_index)->size, cudaMemcpyDefault, d2d));

      copyMetadata(prev, cache->getPrevFieldAt(hit_index));
      GPUZIP_CUDA_CHECK(cudaMemcpyAsync(
          prev->data, cache->getPrevFieldAt(hit_index)->data,
          cache->getPrevFieldAt(hit_index)->size, cudaMemcpyDefault, d2d));

      // We need to make sure client will have the data in hands
      NvtxPush("WAIT_D2D");
      cudaStreamSynchronize(d2d);
      NvtxPop();

      cache->getIndex(timestep, 0);

      GPUZIPLogger::Debug("[RETRIEVE_HIT] After Hit: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      NvtxPop();
    } else {
      NvtxPush("CACHE_MISS");

      actualMissesCount++;

      GPUZIPLogger::Debug("[RETRIEVE_MISS] ===> Cache Miss: timestep %d.\n",
                          timestep);
      GPUZIPLogger::Debug("Before Miss: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      if (pool.timestep[pool.top] != timestep) {
        GPUZIPLogger::Error(
            "[RETRIEVE_MISS] I got timestep = %i and in the pool I found "
            "%i at position %i\n",
            timestep, pool.timestep[pool.top], hit_index);
        exit(1);
      }

      cache->push(timestep, curr, prev, 0);

      cudaStream_t pci = cache->getPciTop();
      cudaStream_t d2d = cache->getD2DTop();

      // It is not in local cache, so bring it from Host Mem
      copyMetadata(curr, &(pool.currs[pool.top]));
      GPUZIP_CUDA_CHECK(cudaMemcpyAsync(curr->data, pool.currs[pool.top].data,
                                        pool.currs[pool.top].size,
                                        cudaMemcpyDefault, pci));

      copyMetadata(prev, &(pool.prevs[pool.top]));
      GPUZIP_CUDA_CHECK(cudaMemcpyAsync(prev->data, pool.prevs[pool.top].data,
                                        pool.prevs[pool.top].size,
                                        cudaMemcpyDefault, pci));

      GPUZIP_CUDA_CHECK(cudaStreamSynchronize(pci));

      if (cache->getTimestepTop() != timestep) {
        GPUZIPLogger::Error(
            "[RETRIEVE_MISS] I got timestep = %i and found %i on top\n",
            timestep, cache->getTimestepTop());
        exit(1);
      }

      GPUZIP_CUDA_CHECK(cudaMemcpyAsync(cache->getCurrTop()->data, curr->data,
                                        curr->size, cudaMemcpyDefault, d2d));

      GPUZIP_CUDA_CHECK(cudaMemcpyAsync(cache->getPrevTop()->data, prev->data,
                                        prev->size, cudaMemcpyDefault, d2d));

      GPUZIPLogger::Debug("[RETRIEVE_MISS] After Miss: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      NvtxPop();
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
                        Compressor<void, void> *compcurr,
                        Compressor<void, void> *compprev) {

    alignPool(timestep);
    int hit_index = cache->findIndex(timestep);

    if (hit_index > -1) {
      NvtxPush("CACHE_HIT");

      hitsCount++;

      GPUZIPLogger::Debug(
          "[RETRIEVE_HIT] Cache Hit: timestep %d in local cache\n", timestep);
      GPUZIPLogger::Debug("Before Hit: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      if (cache->timestepAt(hit_index) != timestep) {
        GPUZIPLogger::Error(
            "[RETRIEVE_HIT] I got timestep = %i and found %i at position %i\n",
            timestep, cache->timestepAt(hit_index), hit_index);
        exit(1);
      }

      cudaStream_t pci = cache->getPciAt(hit_index);
      cudaStream_t d2d = cache->getD2DAt(hit_index);

      if (cache->isPrefetched(hit_index)) {
        prefetchCount++;
        // wait for prefetching action get completed in case it still
        // transfering.. fprintf(stderr, "[RETRIEVE] timestep %i was prefetched.
        // Waiting for it.", timestep);

        NvtxPush("WAIT_PCI_PREFETCH");
        cudaStreamSynchronize(pci);
        NvtxPop();
      } else {
        GPUZIPLogger::Debug(
            "[RETRIEVE_HIT] timestep %i was NOT prefetched. There's "
            "nothing to "
            "wait for.\n",
            timestep);
      }

      copyMetadata(curr, cache->getCurrFieldAt(hit_index));
      curr->size = curr->n1 * curr->n2 * curr->n3 * sizeof(float);
      compcurr->Decompress(cache->getCurrFieldAt(hit_index)->data, curr->data);

      copyMetadata(prev, cache->getPrevFieldAt(hit_index));
      prev->size = prev->n1 * prev->n2 * prev->n3 * sizeof(float);
      compprev->Decompress(cache->getPrevFieldAt(hit_index)->data, prev->data);

      cache->getIndex(timestep, 0);

      GPUZIPLogger::Debug("[RETRIEVE_HIT] After Hit: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      NvtxPop();
    } else {
      NvtxPush("CACHE_MISS");

      actualMissesCount++;

      // It is not in local cache, so bring it from Host Mem
      GPUZIPLogger::Debug("[RETRIEVE_MISS] Cache Miss: timestep %d.\n",
                          timestep);
      GPUZIPLogger::Debug("[RETRIEVE_MISS] Before Miss: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      if (pool.timestep[pool.top] != timestep) {
        GPUZIPLogger::Error(
            "[RETRIEVE_MISS] I got timestep = %i and in the pool I found "
            "%i at position %i\n",
            timestep, pool.timestep[pool.top], hit_index);
        exit(1);
      }

      cache->push(timestep, curr, prev, 0);

      cudaStream_t pci = cache->getPciTop();
      cudaStream_t d2d = cache->getD2DTop();

      if (cache->getTimestepTop() != timestep) {
        GPUZIPLogger::Error(
            "[RETRIEVE_MISS] I got timestep = %i and found %i on top\n",
            timestep, cache->getTimestepTop());
        exit(1);
      }

      GPUZIP_CUDA_CHECK(
          cudaMemcpyAsync(cache->getCurrTop()->data, pool.currs[pool.top].data,
                          pool.currs[pool.top].size, cudaMemcpyDefault, pci));

      GPUZIP_CUDA_CHECK(
          cudaMemcpyAsync(cache->getPrevTop()->data, pool.prevs[pool.top].data,
                          pool.prevs[pool.top].size, cudaMemcpyDefault, pci));

      NvtxPush("WAIT_PCI_MISS");
      cudaStreamSynchronize(pci);
      NvtxPop();

      copyMetadata(curr, cache->getCurrTop());
      curr->size = curr->n1 * curr->n2 * curr->n3 * sizeof(float);
      compcurr->Decompress(cache->getCurrTop()->data, curr->data);

      copyMetadata(prev, cache->getPrevTop());
      prev->size = prev->n1 * prev->n2 * prev->n3 * sizeof(float);
      compprev->Decompress(cache->getPrevTop()->data, prev->data);

      // Waiting for default stream since compressors are using it.
      cudaStreamSynchronize(cudaStreamDefault);

      GPUZIPLogger::Debug("[RETRIEVE_MISS] After Miss: ");
      if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
        cache->print();
      }

      NvtxPop();
    }

    return true;
  }

  /**
   * @brief prefetch prefetches if necessary for a given iteration
   *
   * @param it_revolve The iteration.
   * @return True if at least one prefetch was done, false otherwise.
   */
  virtual bool prefetch(int it_revolve) {
    bool prefetchDone = false;
    for (int i = iterator; i < pav.iter.size(); i++) {
      if (pav.iter[i] == it_revolve) {
        NvtxPush("GPUZIP_PREFETCH_DISPATCH");

        unsigned timestep = pav.timestep[i];

        GPUZIPLogger::Debug("Before Prefetch: ts=%i ", timestep);
        if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
          cache->print();
        }

        // Get index for the timestep in host pool. This points to fpcurr.
        unsigned pool_idx = getSlot(timestep);

        if (pool_idx == pool.size || pool.timestep[pool_idx] != timestep) {
          GPUZIPLogger::Error("ERRO!!\n");
          exit(1);
        }

        GPUZIPLogger::Debug(
            "\n===> Prefetching: pool_idx=%i , ts=%i. It will be used in "
            "iteration: %i. Distance: %i\n",
            pool_idx, timestep, pav.whenused[i], pav.whenused[i] - it_revolve);

        int pos = pav.pos[i];
        cache->insertAt(pav.timestep[i], &(pool.currs[pool_idx]),
                        &(pool.prevs[pool_idx]), pos, true, 0);

        cudaStream_t pci = cache->getPciAt(pos);

        GPUZIP_CUDA_CHECK(cudaMemcpyAsync(
            cache->getCurrFieldAt(pos)->data, pool.currs[pool_idx].data,
            pool.currs[pool_idx].size, cudaMemcpyDefault, pci));

        GPUZIP_CUDA_CHECK(cudaMemcpyAsync(
            cache->getPrevFieldAt(pos)->data, pool.prevs[pool_idx].data,
            pool.prevs[pool_idx].size, cudaMemcpyDefault, pci));

        iterator++;
        prefetchDone = true;

        GPUZIPLogger::Debug("After Prefetch: ts=%i ", timestep);
        if (GPUZIPLogger::GetLevel() == GPUZIPLogger::DEBUG) {
          cache->print();
        }

#ifdef USE_NVTX
        nvtxRangePop();
#endif
      } else if (pav.iter[i] > it_revolve) {
        break;
      }
    }

    return prefetchDone;
  }

  /**
   * @brief free frees allocated memory for streams, pool, and cache.
   */
  void free() {
    cudaStreamDestroy(streams.cache2host);

    for (unsigned p = 0; p < pool.size; p++) {
      if (pool.currs[p].data != NULL) {
        GPUZIP_CUDA_CHECK(cudaFreeHost(pool.currs[p].data));
      }
      if (pool.prevs[p].data != NULL) {
        GPUZIP_CUDA_CHECK(cudaFreeHost(pool.prevs[p].data));
      }
    }

    if (pool.currs != NULL) {
      GPUZIP_CUDA_CHECK(cudaFreeHost(pool.currs));
    }

    if (pool.prevs != NULL) {
      GPUZIP_CUDA_CHECK(cudaFreeHost(pool.prevs));
    }

    if (pool.timestep != NULL) {
      GPUZIP_CUDA_CHECK(cudaFreeHost(pool.timestep));
    }

    cache->free();
  }

public:
  /**
   * @brief Constructs a Prefetch instance.
   *
   * Initializes the prefetching mechanism with the specified parameters.
   *
   * @param numSnaps Number of snapshots to manage.
   * @param timesteps Total number of timesteps.
   * @param biggest_checkpoint_len Maximum allowed length for a field,
   * calculated as n1 * n2 * n3 * sizeof(float).
   * @param chkpt Pointer to the checkpointing mechanism.
   * @param _cache_capacity The capacity of the prefetch cache.
   */
  Prefetch(int numSnaps, unsigned timesteps, size_t biggest_checkpoint_len,
           Checkpointing *chkpt, int _cache_capacity)
      : snaps(numSnaps), timesteps(timesteps),
        biggest_checkpoint_len(biggest_checkpoint_len), chkpt(chkpt),
        cache_capacity(_cache_capacity) {

    int dev;
    GPUZIP_CUDA_CHECK(cudaGetDevice(&dev));
    GPUZIPLogger::PerfTrace("-------------------------------------------\n");
    GPUZIPLogger::PerfTrace("[DEVICE=%i] GPUZIP Memory Allocation\n", dev);

    /*
     * Cache
     */
    cache = new GPUCheckpointCache(cache_capacity, biggest_checkpoint_len);

    /**
     * Streams
     */
    cudaStreamCreate(&streams.cache2host);

    /**
     * Checkpoint pool (host)
     */
    pool.size = numSnaps;
    pool.top = -1;
    GPUZIP_CUDA_CHECK(
        cudaMallocHost(&(pool.currs), numSnaps * sizeof(Field_t)));
    GPUZIP_CUDA_CHECK(
        cudaMallocHost(&(pool.prevs), numSnaps * sizeof(Field_t)));
    GPUZIP_CUDA_CHECK(cudaMallocHost(&(pool.timestep), numSnaps * sizeof(int)));

    for (unsigned p = 0; p < pool.size; p++) {
      pool.timestep[p] = -1;
      GPUZIP_CUDA_CHECK(
          cudaMallocHost(&(pool.currs[p].data), biggest_checkpoint_len));
      GPUZIP_CUDA_CHECK(
          cudaMallocHost(&(pool.prevs[p].data), biggest_checkpoint_len));
    }

    GPUZIPLogger::PerfTrace(
        "[GPUZIP_RESOURCES][DEVICE=%i] GPU Cache Size = %i positions\n", dev,
        cache->getCapacity());
    GPUZIPLogger::PerfTrace(
        "[GPUZIP_RESOURCES][DEVICE=%i] Checkpoint Pool Size: %i\n", dev,
        pool.size);
    GPUZIPLogger::PerfTrace(
        "[GPUZIP_RESOURCES][DEVICE=%i] Field max len: %lu (MB)\n", dev,
        (double)biggest_checkpoint_len / 1000000.0);
    GPUZIPLogger::PerfTrace(
        "[MEM_TRACK][GPU][DEVICE=%i][GPUZIP__CACHE] %.2lf (MB)\n", dev,
        ((double)((biggest_checkpoint_len + biggest_checkpoint_len) *
                  cache->getCapacity())) /
            1000000.0);
    GPUZIPLogger::PerfTrace(
        "[MEM_TRACK][HOST][DEVICE=%i][GPUZIP__CHECKPOINT_POOL] %.2lf (GB)\n",
        dev,
        (double)((biggest_checkpoint_len + biggest_checkpoint_len) *
                 pool.size) /
            1000000000.0);
  }

  void Report() {
    fprintf(stderr, "==============GPUZIP PREFETCH REPORT=================\n");
    fprintf(stderr, "Number of hits (prefetch+hits): %i \n", hitsCount);
    fprintf(stderr, "Number of unavoidable misses: %i \n", actualMissesCount);
    fprintf(stderr, "Number of prefetched checkpoints: %i \n", prefetchCount);
    fprintf(stderr, "=====================================================\n");
  }

  /**
   * @brief Initializes the GPUZIP prefetching system.
   *
   * This method performs the necessary setup steps before prefetching
   * operations can begin. It includes logging, device retrieval, checkpoint
   * resets, and algorithm-specific prefetch setup.
   *
   * @note This function should be called before starting prefetching
   * operations.
   */
  virtual void Setup() {
    int dev;
    GPUZIP_CUDA_CHECK(cudaGetDevice(&dev));
    GPUZIPLogger::PerfTrace("===================================\n");
    GPUZIPLogger::PerfTrace("[DEVICE=%i] Starting GPUZIP Setup\n", dev);

    NvtxPush("GPUZIP_SETUP");

    reset();
    PrefetchSetupAlgorithm();
    chkpt->Reset();

#ifdef USE_NVTX
    nvtxRangePop();
#endif

    size_t pav_size =
        (pav.iter.size() * sizeof(int)) + (pav.timestep.size() * sizeof(int));
    GPUZIPLogger::PerfTrace("[MEM_TRACK][HOST][DEVICE=%i][PAV] %lu (bytes)\n",
                            dev, pav_size);
    GPUZIPLogger::PerfTrace("[GPUZIP_RESOURCES][DEVICE=%i] PAV Len = %lu\n",
                            dev, pav.iter.size());
    GPUZIPLogger::PerfTrace("[DEVICE=%i] Complete GPUZIP Setup\n", dev);
    GPUZIPLogger::PerfTrace("===================================\n");
  }

  /**
   * @brief Saves a checkpoint at the specified timestep.
   *
   * This method stores the current and previous field data at a given timestep,
   * optionally applying compression. The saved checkpoint can later be used
   * for restoration or prefetching.
   *
   * @param timestep The simulation timestep at which the checkpoint is saved.
   * @param curr Pointer to the current field data.
   * @param prev Pointer to the previous field data.
   * @param compcurr Pointer to the compressor used for compressing `curr` (if
   * applicable).
   * @param compprev Pointer to the compressor used for compressing `prev` (if
   * applicable).
   *
   * @return `true` if the checkpoint was successfully saved, `false` otherwise.
   *
   * @note The function may utilize compression if the compressor objects are
   * provided. The saved checkpoint can be used for rollback or prefetching
   * strategies.
   */
  virtual bool Save(unsigned timestep, Field_t *curr, Field_t *prev,
                    Compressor<void, void> *compcurr,
                    Compressor<void, void> *compprev) {
#ifdef USE_NVTX
    compcurr->EnableProfile();
    compprev->EnableProfile();
    NvtxPush("GPUZIP_SAVE");
#endif

    bool ret = save(timestep, curr, prev, compcurr, compprev);

    NvtxPop();

    return ret;
  }

  /**
   * @brief Saves a checkpoint at the specified timestep.
   *
   * This method stores the current and previous field data at a given timestep.
   * The saved checkpoint can later be used for rollback, recovery, or
   * prefetching.
   *
   * @param timestep The simulation timestep at which the checkpoint is saved.
   * @param curr Pointer to the current field data.
   * @param prev Pointer to the previous field data.
   *
   * @return `true` if the checkpoint was successfully saved, `false` otherwise.
   *
   * @note This variant of `Save` does not use compression.
   */
  virtual bool Save(int timestep, Field_t *curr, Field_t *prev) {

    NvtxPush("GPUZIP_SAVE");

    bool ret = save(timestep, curr, prev);

    NvtxPop();

    return ret;
  }

  /**
   * @brief Retrieves a checkpoint for the specified timestep.
   *
   * This method loads the checkpointed field data corresponding to the given
   * timestep. The retrieved data is stored in the provided `curr` and `prev`
   * pointers, allowing rollback or restoration of simulation states.
   *
   * @param timestep The simulation timestep to retrieve the checkpoint for.
   * @param curr Pointer to store the retrieved current field data.
   * @param prev Pointer to store the retrieved previous field data.
   *
   * @return `true` if the checkpoint was successfully retrieved, `false`
   * otherwise.
   *
   * @note This variant of `Retrieve` does not use decompression.
   */
  virtual bool Retrieve(int timestep, Field_t *curr, Field_t *prev) {
    NvtxPush("GPUZIP_RETRIEVE");
    bool ret = retrieve(timestep, curr, prev, streams.cache2host);
    NvtxPop();
    return ret;
  }

  /**
   * @brief Retrieves a checkpoint for the specified timestep using a CUDA
   * stream.
   *
   * This method loads the checkpointed field data corresponding to the given
   * timestep, storing it in the provided `curr` and `prev` pointers. It
   * utilizes the provided CUDA stream to enable asynchronous execution.
   *
   * @param timestep The simulation timestep to retrieve the checkpoint for.
   * @param curr Pointer to store the retrieved current field data.
   * @param prev Pointer to store the retrieved previous field data.
   * @param stream The CUDA stream to execute the retrieval asynchronously.
   *
   * @return `true` if the checkpoint was successfully retrieved, `false`
   * otherwise.
   *
   * @note This variant of `Retrieve` allows for concurrent execution with other
   * GPU tasks.
   */
  virtual bool Retrieve(int timestep, Field_t *curr, Field_t *prev,
                        cudaStream_t stream) {
    NvtxPush("GPUZIP_RETRIEVE");
    bool ret = retrieve(timestep, curr, prev, stream);
    NvtxPop();
    return ret;
  }

  /**
   * @brief Retrieves a checkpoint for the specified timestep with
   * decompression.
   *
   * This method loads checkpointed field data for a given timestep, storing it
   * in the provided `curr` and `prev` pointers. If compression is enabled, it
   * uses the provided `compcurr` and `compprev` decompressors to restore the
   * data before storing it.
   *
   * @param timestep The simulation timestep to retrieve the checkpoint for.
   * @param curr Pointer to store the retrieved current field data.
   * @param prev Pointer to store the retrieved previous field data.
   * @param compcurr Pointer to the decompressor for the `curr` field.
   * @param compprev Pointer to the decompressor for the `prev` field.
   *
   * @return `true` if the checkpoint was successfully retrieved, `false`
   * otherwise.
   *
   * @note This variant of `Retrieve` is used when checkpoint compression is
   * enabled.
   */
  virtual bool Retrieve(unsigned timestep, Field_t *curr, Field_t *prev,
                        Compressor<void, void> *compcurr,
                        Compressor<void, void> *compprev) {
#ifdef USE_NVTX
    compcurr->EnableProfile();
    compprev->EnableProfile();
    NvtxPush("GPUZIP_RETRIEVE");
#endif

    bool ret = retrieve(timestep, curr, prev, compcurr, compprev);

    NvtxPop();

    return ret;
  }

  /**
   * @brief Determines the checkpointing action for a given iteration.
   *
   * This method processes the specified iteration and determines the
   * appropriate checkpointing action (e.g., saving, restoring, advancing,
   * etc.). It ensures that the checkpointing strategy follows the predefined
   * scheduling rules.
   *
   * @param it The current iteration number.
   * @return `true` if a checkpointing action was successfully dispatched,
   *         `false` otherwise.
   *
   * @note This method is central to managing checkpointing operations across
   *       simulation iterations.
   */
  virtual bool Dispatch(int it) {
    bool ret = prefetch(it);
    return ret;
  }

  /**
   * @brief Frees the resources used by the checkpointing system.
   *
   * This method releases memory and other resources allocated for checkpointing
   * operations, including clearing caches, resetting internal variables, and
   * deallocating any allocated structures. It ensures that all resources are
   * cleaned up before the checkpointing system is terminated or reinitialized.
   */
  void Free() {
    NvtxPush("GPUZIP_FREE");
    free();
    NvtxPop();
  }
};
