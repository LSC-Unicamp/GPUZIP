/**
 * structs.h
 * @brief Defines structures used in the Prefetching API.
 * 
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date November 25, 2023
 */
#pragma once
#include <iostream>
#include <vector>

/**
 * @brief Structure representing CUDA streams for saving and retrieving data.
 */
typedef struct {
  cudaStream_t cache2host;     ///< CUDA stream for saving data.
  cudaStream_t retrieve; ///< CUDA stream for retrieving data.
} Streams_t;

typedef struct {
  void *data;  ///< Pointer to the data.
  size_t size; ///< Size of the data in bytes.
  size_t n1;   ///< Dimension 1 size.
  size_t n2;   ///< Dimension 2 size.
  size_t n3;   ///< Dimension 3 size.
} Field_t;

/**
 * @brief Structure representing an action vector for prefetching.
 */
typedef struct {
  std::vector<int> timestep; ///< Array storing timesteps to be prefetched.
  std::vector<int> iter;     ///< Array storing iteration numbers for prefetches.
  std::vector<int> whenused; ///< Array storing when (iteration) the checkpoint will be used after prefetched.
  std::vector<int> pos;
} PrefetchAction_t;

/**
 * @brief Structure representing a checkpoint pool.
 */
typedef struct {
  // The following fields store checkpoint pool data structure
  int top = -1;    ///< Top index of the pool.
  unsigned size;  ///< Size of the pool.
  int *order; /// 

  // The following fields store data concerned to checkpoint
  int *timestep;  ///< Array storing timesteps in the pool.
  Field_t *currs; ///< Array storing current field data in the pool.
  Field_t *prevs; ///< Array storing previous field data in the pool.
} ChkptPool_t;
