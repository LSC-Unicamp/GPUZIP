#include "GPUCheckpointCache.cuh"
#include <unordered_map>

/**
 * @class GPUCheckpointCacheMemento
 * @brief Manages snapshots of GPUCheckpointCache states, allowing save,
 * restore, and reset operations.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date July 2nd, 2024
 */
class GPUCheckpointCacheMemento {
private:
  std::unordered_map<int, GPUCheckpointCache *> history;

public:
  /**
   * @brief Constructor. Initializes an empty GPUCheckpointCacheMemento.
   */
  GPUCheckpointCacheMemento() {}

  /**
   * @brief Destructor. Cleans up all saved checkpoints in the history.
   */
  ~GPUCheckpointCacheMemento() { reset(); }

  /**
   * @brief Copies the state of one GPUCheckpointCache to another.
   * @param src The source GPUCheckpointCache to copy from.
   * @param dest The destination GPUCheckpointCache to copy to.
   *
   * Copies arrays such as `order`, `prefetched`, `timesteps`, and `lastused`,
   * as well as the `length` property, from `src` to `dest`.
   */
  void copy(GPUCheckpointCache *src, GPUCheckpointCache *dest) {
    for (int i = 0; i < src->getCapacity(); i++) {
      dest->order[i] = src->order[i];
      dest->prefetched[i] = src->prefetched[i];
      dest->timesteps[i] = src->timesteps[i];
      dest->lastused[i] = src->lastused[i];
    }
    dest->length = src->length;
  }

  /**
   * @brief Saves the current state of a GPUCheckpointCache at a specific
   * iteration.
   * @param it The iteration number to associate with this checkpoint.
   * @param cache The GPUCheckpointCache instance to save.
   *
   * If a checkpoint already exists for the given iteration, it is overwritten.
   */
  void save(int it, GPUCheckpointCache *cache) {
    if (history.find(it) == history.end()) {
      history.erase(it);
    }

    auto newcache = new GPUCheckpointCache(cache->getCapacity());
    copy(cache, newcache);
    history[it] = std::move(newcache);
  }

  /**
   * @brief Restores a saved state of a GPUCheckpointCache.
   * @param it The iteration number of the checkpoint to restore.
   * @param cache The GPUCheckpointCache instance to restore into.
   *
   * Copies the saved state associated with the specified iteration back into
   * `cache`. Assumes a checkpoint for the given iteration exists.
   */
  void restore(int it, GPUCheckpointCache *cache) { copy(history[it], cache); }

  /**
   * @brief Resets the history by clearing all saved checkpoints.
   *
   * Releases memory associated with all stored GPUCheckpointCache instances.
   */
  void reset() { history.clear(); }
};
