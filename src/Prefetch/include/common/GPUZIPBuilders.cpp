#pragma once
#include "GPUZIPConfig.h"

#include "GPUZIPLogger.cpp"
#include "checkpointing/revolve/RevolveCheckpointing.cpp"
#include "checkpointing/trace/TraceCheckpointing.cpp"
#include "prefetch/CheckpointOnly.cuh"
#include "prefetch/Prefetch.cuh"
#include "common/real_t.h"

#ifdef CUSZP
#include "CompressorCuSZp.hpp"
#endif

#ifdef NVCOMP_BITCOMP
#include "CompressorBitcomp.hpp"
#endif

#ifdef ZFP
#include "CompressorZFP.hpp"
#endif

/**
 * @class GPUZIPBuilders
 * @brief A factory class responsible for constructing various GPUZIP
 * components, including checkpointing, prefetching, and compression mechanisms.
 *
 * This class provides static methods to:
 * - Log the current configuration settings.
 * - Build the appropriate checkpointing strategy (Trace or Revolve).
 * - Build the correct prefetching mechanism.
 * - Construct the appropriate compressor (cuZFP, Bitcomp, cuSZp).
 *
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Feb 25, 2024
 */
class GPUZIPBuilders {
public:
  /**
   * @brief Constructs the appropriate prefetching mechanism.
   *
   * @param gpuzip_config Pointer to the GPUZIP configuration structure.
   * @param n1 Grid dimension 1.
   * @param n2 Grid dimension 2.
   * @param n3 Grid dimension 3.
   * @param steps Number of simulation steps.
   * @param chkpt Pointer to the Checkpointing object.
   * @return A pointer to a Prefetch or CheckpointOnly object.
   */
  static Prefetch *PrefetchBuilder(const gpuzip_config_t *gpuzip_config,
                                   size_t n1, size_t n2, size_t n3, int steps,
                                   Checkpointing *chkpt) {

    size_t biggestCheckpointSize;

    if (gpuzip_config->compressor > 0) {
      if (gpuzip_config->compression_factor > 0) {
        biggestCheckpointSize = (size_t)((double)(n1 * n2 * n3 * sizeof(real_t)) / gpuzip_config->compression_factor);
      } else {
        auto comp = CompressorBuilder(gpuzip_config, n1, n2, n3);
        biggestCheckpointSize = comp.get()->CompressedMaxSize();
      }
    } else {
      biggestCheckpointSize = n1 * n2 * n3 * sizeof(real_t);
    }

    if (gpuzip_config->cache_capacity >= 2) {
      GPUZIPLogger::Info("Using Prefetching Algorithm.\n");
      return new Prefetch(chkpt->GetNumberOfCheckpoints(), steps,
                          biggestCheckpointSize, chkpt,
                          gpuzip_config->cache_capacity);
    } else {
      GPUZIPLogger::Info("Using Checkpointing Only (no prefetching).\n");
      return new CheckpointOnly(chkpt->GetNumberOfCheckpoints(), steps,
                                biggestCheckpointSize, chkpt, gpuzip_config->compressor > 0);
    }
  }

  /**
   * @brief Constructs the appropriate prefetching mechanism.
   *
   * @param gpuzip_config Pointer to the GPUZIP configuration structure.
   * @param biggestCheckpointSize The size of checkpoints in bytes
   * @param steps Number of simulation steps.
   * @param chkpt Pointer to the Checkpointing object.
   * @return A pointer to a Prefetch or CheckpointOnly object.
   */
  static Prefetch *PrefetchBuilder(const gpuzip_config_t *gpuzip_config,
                                   size_t biggestCheckpointSize, int steps,
                                   Checkpointing *chkpt) {
    if (gpuzip_config->cache_capacity >= 2) {
      GPUZIPLogger::Info("Using Prefetching Algorithm.\n");
      return new Prefetch(chkpt->GetNumberOfCheckpoints(), steps,
                          biggestCheckpointSize, chkpt,
                          gpuzip_config->cache_capacity);
    } else {
      GPUZIPLogger::Info("Using Checkpointing Only (no prefetching).\n");
      return new CheckpointOnly(chkpt->GetNumberOfCheckpoints(), steps,
                                biggestCheckpointSize, chkpt);
    }
  }

  /**
   * @brief Constructs the appropriate checkpointing strategy.
   *
   * @param gpuzip_config Pointer to the configuration structure.
   * @param steps Number of simulation steps.
   * @return A pointer to a Checkpointing object.
   */
  static Checkpointing *
  CheckpointingBuilder(const gpuzip_config_t *gpuzip_config, int steps) {
    if (gpuzip_config->checkpointing_algorithm == 0) {
      GPUZIPLogger::Info("Using Trace Checkpointing (%s).\n",
                         gpuzip_config->trace_file_path);
      return new TraceCheckpointing(steps, gpuzip_config->trace_file_path);
    } else {
      GPUZIPLogger::Info("Using Revolve Checkpointing .\n");
      return new RevolveCheckpointing(steps, gpuzip_config->revolve_log_level);
    }
  }

  /**
   * @brief Constructs the appropriate compressor based on the configuration.
   *
   * @param gpuzip_config Pointer to the configuration structure.
   * @param n1 Grid dimension 1.
   * @param n2 Grid dimension 2.
   * @param n3 Grid dimension 3.
   * @return A unique pointer to the selected compressor.
   */
  static std::unique_ptr<Compressor<void, void>>
  CompressorBuilder(const gpuzip_config_t *gpuzip_config, size_t n1, size_t n2,
                    size_t n3) {
    // # NoCompression=0, cuZFP=1, Bitcomp=2, or cuSZp=3. Integer.
    if (gpuzip_config->compressor == 1) {
#ifdef ZFP
      return std::make_unique<CompressorZFP<void, void>>(
          n1, n2, n3, "float", gpuzip_config->zfp_bit_rate);
#else
      fprintf(stderr, "This software was not built with cuZFP. Rebuild with "
                      "-DZFP=1 or see docs for more information.\n");
      exit(-1);
#endif
    } else if (gpuzip_config->compressor == 2) {
#ifdef NVCOMP_BITCOMP
      return std::make_unique<CompressorBitcomp<void, void>>(
          n1, n2, n3, gpuzip_config->bitcomp_delta_config,
          gpuzip_config->bitcomp_range_fraction,
          gpuzip_config->bitcomp_num_sigma, gpuzip_config->bitcomp_delta,
          "float",
          gpuzip_config->bitcomp_algorithm == 0 ? "default" : "sparse");
#else
      fprintf(stderr,
              "This software was not built with NVIDIA Bitcomp. Rebuild with "
              "-DNVCOMP_BITCOMP=1 or see docs for more information.\n");
      exit(-1);
#endif
    } else if (gpuzip_config->compressor == 3) {
#ifdef CUSZP
      return std::make_unique<CompressorCuSZp<void, void>>(
          n1, n2, n3, gpuzip_config->cuszp_err_bound);
#else
      fprintf(stderr,
              "This software was not built with NVIDIA Bitcomp. Rebuild "
              "with -DCUSZP=1 or see docs for more information.\n");
      exit(-1);
#endif
    } else if (gpuzip_config->compressor != 0) {
      fprintf(stderr, "Compressor is misconfigured.\n");
      exit(-1);
    }

    return nullptr;
  }
};
