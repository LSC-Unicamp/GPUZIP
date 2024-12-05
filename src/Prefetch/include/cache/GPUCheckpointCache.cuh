#pragma once

#include "../common/cudaUtils.cuh"
#include "../common/structs.h"
#include <memory>
#include <unordered_map>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

/**
 * @class GPUCheckpointCache
 * @brief Defines structures of the cache used in the Prefetching API.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date July 2nd, 2024
 */
class GPUCheckpointCache {
protected:
  int length;
  int capacity;
  int *order; // [3,2,0,1] means the MRU is the timestep[3].
  int *timesteps;
  int *lastused;
  Field_t *currs;
  Field_t *prevs;
  bool *prefetched;
  cudaStream_t *pci;
  cudaStream_t *d2d;
  size_t maxChkptLen;

  friend class GPUCheckpointCacheMemento;

  /**
   * @brief Copies the contents of one Field_t object to another.
   *
   * This method copies the size and other attributes from the source Field_t
   * object to the destination Field_t object. It assumes that both the
   * destination and source Field_t pointers are valid and properly initialized.
   *
   * @param dest Pointer to the destination Field_t object where the contents
   * will be copied.
   * @param src Pointer to the source Field_t object from which the contents
   * will be copied.
   *
   * @note Both `dest` and `src` must be valid, non-null pointers to properly
   * initialized Field_t objects.
   */
  virtual void copyField(Field_t *dest, Field_t *src) {
    dest->size = src->size;
    dest->n1 = src->n1;
    dest->n2 = src->n2;
    dest->n3 = src->n3;
  }

  /**
   * @brief Sets the length of the cache.
   *
   * This method sets the length of the cache to the specified value.
   * If the specified length exceeds the capacity of the cache, an error
   * message is printed and the program exits with a failure status.
   *
   * @param len The desired length of the cache.
   *
   * @throws This function has a sanity check if len > capacity. It will print
   * an error message and terminate the program.
   */
  void setLength(int len) {
    if (len > capacity) {
      fprintf(stderr,
              "PREFETCHAPI_ERR: Invalid Length. len=%i > capacity=%i.\n", len,
              capacity);
      exit(1);
    }
    this->length = len;
  }

  /**
   * @brief Checks if the specified position is within the valid range of the
   * cache.
   *
   * This method checks whether the given position is within the valid range of
   * the cache's length. If the position is out of range, an error message is
   * printed and the program exits with a failure status.
   *
   * @param pos The position to check within the cache.
   *
   * @throws This function contains a sanity check if pos is out of range. It
   * will print an error message and terminate the program.
   */
  void checkPos(int pos) {
    if (pos > getLength() - 1) {
      fprintf(stderr, "PREFETCHAPI_ERR: Out of range pos=%i, length=%i\n", pos,
              getLength());
      exit(1);
    }
  }

  /**
   * @brief Returns data index
   */
  int lruIndex() { return order[getLength() - 1]; }

  /**
   * @brief Receives the data index
   *
   * @param dataIndex The position located the data in the data arrays such as
   * timesteps, currs, prevs, streams etc
   */
  void promoteMru(int dataIndex) {
    for (int i = 0; i < getLength(); ++i) {
      if (order[i] == dataIndex) {
        for (int j = i; j > 0; --j) {
          order[j] = order[j - 1];
        }
        order[0] = dataIndex;
        break;
      }
    }
  }

  /**
   * @brief Reorders the cache by moving the specified position to the front.
   *
   * This method shifts all elements in the `order` array one position to the
   * right and places the specified new position at the front. This can be used
   * to mark the specified position as the most recently used.
   *
   * @param dataIndex The position located the data in the data arrays such as
   * timesteps, currs, prevs, streams etc
   */
  void reorder(int dataIndex) {
    for (int i = getCapacity() - 1; i > 0; --i) {
      order[i] = order[i - 1];
    }
    order[0] = dataIndex;
  }

public:
  /**
   * @brief Constructs a GPUCheckpointCache with a specified capacity. This
   * constructor also allocates memory in GPU (for the given maxChkptLen)
   *
   * @param capacity The capacity of the cache.
   * @param maxChkptLen Optional. The maximum checkpoint length in bytes.
   * Defaults to 0 -- for dry runs.
   */
  GPUCheckpointCache(int capacity, size_t maxChkptLen = 0) {
    this->capacity = capacity;
    this->order = (int *)malloc(capacity * sizeof(int));
    this->timesteps = (int *)malloc(capacity * sizeof(int));
    this->currs = (Field_t *)malloc(capacity * sizeof(Field_t));
    this->prevs = (Field_t *)malloc(capacity * sizeof(Field_t));
    this->prefetched = (bool *)malloc(capacity * sizeof(bool));
    this->lastused = (int *)malloc(capacity * sizeof(int));
    this->length = 0;
    this->maxChkptLen = maxChkptLen;

    if (maxChkptLen > 0) {
      GPUZIP_CUDA_CHECK(cudaMallocHost(&(currs), capacity * sizeof(Field_t)));
      GPUZIP_CUDA_CHECK(cudaMallocHost(&(prevs), capacity * sizeof(Field_t)));
      GPUZIP_CUDA_CHECK(
          cudaMallocHost(&(pci), capacity * sizeof(cudaStream_t)));
      GPUZIP_CUDA_CHECK(
          cudaMallocHost(&(d2d), capacity * sizeof(cudaStream_t)));
    }

    for (int i = 0; i < capacity; i++) {
      timesteps[i] = -1;
      order[i] = -1;
      lastused[i] = -1;
      prefetched[i] = false;

      if (maxChkptLen > 0) {
        cudaStreamCreate(&pci[i]);
        cudaStreamCreate(&d2d[i]);
        GPUZIP_CUDA_CHECK(cudaMalloc(&(currs[i].data), maxChkptLen));
        GPUZIP_CUDA_CHECK(cudaMalloc(&(prevs[i].data), maxChkptLen));
      }
    }
  }

  ~GPUCheckpointCache() {
    free();
  }

  /**
   * @brief Returns the current length of the cache.
   *
   * @return The current length of the cache.
   */
  int getLength() { return this->length; }

  /**
   * @brief Returns the capacity of the cache.
   *
   * @return The capacity of the cache.
   */
  int getCapacity() { return this->capacity; }

  /**
   * @brief Clears the cache by resetting all its elements.
   */
  void clear() {
    for (int i = 0; i < capacity; i++) {
      timesteps[i] = -1;
      order[i] = -1;
      lastused[i] = -1;
      prefetched[i] = false;
    }
    setLength(0);
  }

  /**
   * @brief Frees up all memory allocated by the cache.
   */
  void free() {
    if (maxChkptLen > 0) {
      for (int i = 0; i < capacity; i++) {
        GPUZIP_CUDA_CHECK(cudaFree(currs[i].data));
        GPUZIP_CUDA_CHECK(cudaFree(prevs[i].data));
        GPUZIP_CUDA_CHECK(cudaStreamDestroy(d2d[i]));
        GPUZIP_CUDA_CHECK(cudaStreamDestroy(pci[i]));
      }
    }
  }

  /**
   * @brief Console prints the freshest status of the cache
   */
  void print() {
    fprintf(stderr, "[");
    for (int i = 0; i < getLength(); i++) {
      fprintf(stderr, "%i=%i", timestepAt(i), lastUsedAt(i));
      if (i < getLength() - 1) {
        fprintf(stderr, ",");
      }
    }
    fprintf(stderr, "]\n");
  }

  /**
   * @brief Retrieves the timestep at the specified position in the cache.
   *
   * @param pos The position in the cache for which to retrieve the timestep.
   * @return The timestep at the specified position.
   *
   * @throws This function will terminate the program if the position is out of
   * range, as determined by the `checkPos` method.
   */
  int timestepAt(int pos) {
    checkPos(pos);
    return timesteps[order[pos]];
  }

  /**
   * @brief Checks if the cache is full.
   *
   * @return `true` if the cache is full (i.e., the length equals the
   * capacity), `false` otherwise.
   */
  bool isFull() { return this->getLength() == this->getCapacity(); }

  /**
   * @brief Adds a new item to the cache, either by replacing the least
   * recently used item or by adding the item to the next available slot.
   *
   * This method adds a new timestep and associated `Field_t` data (`curr` and
   * `prev`) to the cache. If the cache is full, it sacrifices the least
   * recently used (LRU) item, replaces it with the new data, and updates the
   * order to reflect the new addition. If the cache is not full, it simply
   * adds the new item to the next available position and updates the length of
   * the cache.
   *
   * @param timestep The timestep to be added to the cache.
   * @param curr Pointer to the current `Field_t` object to be copied into the
   * cache.
   * @param prev Pointer to the previous `Field_t` object to be copied into the
   * cache.
   *
   * @note If the cache is full, the method synchronizes CUDA streams
   * associated with the LRU item to prevent data override before replacing the
   * LRU item.
   */
  void push(int timestep, Field_t *curr, Field_t *prev, int it,
            bool wasPrefetched = false) {
    if (isFull()) {
      // Sacrifice the LRU item.
      // Then set the new data where used to be LRU
      // but it will be the top in the order array
      int lru = lruIndex();

      // Prevents data override
      if (maxChkptLen > 0) {
#ifdef USE_NVTX
        nvtxRangePush("CACHE_PUSH_SYNC");
#endif
        cudaStreamSynchronize(pci[lru]);
        cudaStreamSynchronize(d2d[lru]);
#ifdef USE_NVTX
        nvtxRangePop();
#endif
      }

      reorder(lru);

      if (maxChkptLen > 0) {
        copyField(&currs[lru], curr);
        copyField(&prevs[lru], curr);
      }

      timesteps[lru] = timestep;
      prefetched[lru] = wasPrefetched;
      lastused[lru] = it;
    } else {
      // Add new item to the cache
      timesteps[getLength()] = timestep;
      prefetched[getLength()] = wasPrefetched;
      lastused[getLength()] = it;
      if (maxChkptLen > 0) {
        copyField(&currs[getLength()], curr);
        copyField(&prevs[getLength()], prev);
      }
      reorder(getLength());
      setLength(getLength() + 1);
    }
  }

  /**
   * Inserts into a specific position of the cache (@param pos)
   */
  void insertAt(int timestep, Field_t *curr, Field_t *prev, int pos,
                bool wasPrefetched, int it) {
    if (maxChkptLen > 0) {
#ifdef USE_NVTX
      nvtxRangePush("CACHE_INSERT_AT_SYNC");
#endif
      cudaStreamSynchronize(pci[order[pos]]);
      cudaStreamSynchronize(d2d[order[pos]]);
#ifdef USE_NVTX
      nvtxRangePop();
#endif
    }

    timesteps[order[pos]] = timestep;
    prefetched[order[pos]] = wasPrefetched;
    lastused[order[pos]] = it;
    if (maxChkptLen > 0) {
      copyField(&currs[order[pos]], curr);
      copyField(&prevs[order[pos]], prev);
    }
  }

  /**
   * Gets the index of a given timestep in the cache.
   * Get the index, if > -1, it is a cache hit otherwise cache miss.
   * Notice that when it is a hit, the hit index is promoted to MRU.
   * @param timestep - The given timestep
   * @return index
   */
  int getIndex(int timestep, int it) {
    int hitIndex = findIndex(timestep);

    if (hitIndex > -1) {
      int pos = order[hitIndex];
      promoteMru(pos);
      lastused[pos] = it;
    }

    return hitIndex;
  }

  /**
   * @brief Finds the index of a given timestep in the cache.
   *
   * Get the index, if > -1, it is a cache hit otherwise cache miss.
   * Notice that as opposite of the `getIndex` method no index promotion happes.
   *
   * @param timestep - The given timestep
   * @return index -1 means cache miss.
   */
  int findIndex(int timestep) {
    for (int i = 0; i < getLength(); i++) {
      if (timesteps[order[i]] == timestep) {
        return i;
      }
    }
    return -1;
  }

  /**
   * @brief Same as findIndex but iterates from the last to the first item in
   * the array.
   *
   * @param timestep - The given timestep
   * @return index -1 means cache miss.
   */
  int findLastIndex(int timestep) {
    for (int i = getLength() - 1; i >= 0; --i) {
      if (timesteps[order[i]] == timestep) {
        return i;
      }
    }
    return -1;
  }

  /**
   * @brief Retrieves the current Field_t object at the top position.
   *
   * @return Pointer to the current Field_t object at the top position of the
   * cache.
   */
  Field_t *getCurrTop() { return &currs[order[0]]; }

  /**
   * @brief Retrieves the previous Field_t object at the top position.
   *
   * @return Pointer to the previous Field_t object at the top position of the
   * cache.
   */
  Field_t *getPrevTop() { return &prevs[order[0]]; }

  /**
   * @brief Retrieves the timestep at the top position.
   *
   * @return The timestep at the top position of the cache.
   */
  int getTimestepTop() { return timesteps[order[0]]; }

  /**
   * @brief Retrieves the current Field_t object at the specified position.
   *
   * @param pos The position in the cache.
   * @return Pointer to the current Field_t object at the specified position.
   */
  Field_t *getCurrFieldAt(int pos) { return &currs[order[pos]]; }

  /**
   * @brief Retrieves the previous Field_t object at the specified position.
   *
   * @param pos The position in the cache.
   * @return Pointer to the previous Field_t object at the specified position.
   */
  Field_t *getPrevFieldAt(int pos) { return &prevs[order[pos]]; }

  /**
   * @brief Retrieves the CUDA stream associated with PCI at the top position.
   *
   * @return The CUDA stream associated with PCI at the top position of the
   * cache.
   */
  cudaStream_t getPciTop() { return pci[order[0]]; }

  /**
   * @brief Retrieves the CUDA stream associated with PCI at the specified
   * position.
   *
   * @param pos The position in the cache.
   * @return The CUDA stream associated with PCI at the specified position.
   */
  cudaStream_t getPciAt(int pos) { return pci[order[pos]]; }

  /**
   * @brief Retrieves the CUDA stream associated with D2D at the top position.
   *
   * @return The CUDA stream associated with D2D at the top position of the
   * cache.
   */
  cudaStream_t getD2DTop() { return d2d[order[0]]; }

  /**
   * @brief Retrieves the CUDA stream associated with D2D at the specified
   * position.
   *
   * @param pos The position in the cache.
   * @return The CUDA stream associated with D2D at the specified position.
   */
  cudaStream_t getD2DAt(int pos) { return d2d[order[pos]]; }

  /**
   * @brief Checks if the item at the specified position is prefetched.
   *
   * @param pos The position in the cache.
   * @return `true` if the item at the specified position is prefetched, `false`
   * otherwise.
   */
  bool isPrefetched(int pos) { return prefetched[order[pos]]; }

/**
 * @brief Indicates the last iteration a given @pos were used.
 * 
 * @param pos The position in the cache.
 * @return `integer` the iteration the cache position were used
 */
  int lastUsedAt(int pos) { return lastused[order[pos]]; }
};
