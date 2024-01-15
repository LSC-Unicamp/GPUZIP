/*
 * structs.h
 * Author: Thiago Maltempi <tmaltempi@ic.unicamp.br>
 * Author: Sandro Rigo <srigo@ic.unicamp.br>
 * Date: November 25, 2023
 *
 * Description: Defines structures used in the Prefetching API.
 */
#pragma once
#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief Structure representing CUDA streams for saving and retrieving data.
 */
typedef struct {
  cudaStream_t save;     ///< CUDA stream for saving data.
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
  int timestep[2000]; ///< Array storing timesteps to be prefetched.
  int iter[2000];     ///< Array storing iteration numbers for prefetches.
} PrefetchAction_t;

/**
 * @brief Structure representing a checkpoint pool buffer.
 */
typedef struct {
  int timestep = -1; ///< Timestep associated with the buffer.
  Field_t *curr;     ///< Pointer to the current field data.
  Field_t *prev;     ///< Pointer to the previous field data.
  int top = -1;      ///< Top index of the buffer.
} ChkptPoolBf_t;

/**
 * @brief Structure representing a checkpoint pool.
 */
typedef struct {
  int *timestep;  ///< Array storing timesteps in the pool.
  int top = -1;    ///< Top index of the pool.
  unsigned size;  ///< Size of the pool.
  Field_t *currs; ///< Array storing current field data in the pool.
  Field_t *prevs; ///< Array storing previous field data in the pool.
} ChkptPool_t;
