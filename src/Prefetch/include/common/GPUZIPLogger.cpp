#pragma once
#include "GPUZIPConfig.h"
#include "structs.h"
#include <cstdarg>
#include <cstdio>
#include <iostream>

/**
 * @brief A simple logger utility for GPUZIP with multiple log levels.
 *
 * Provides logging methods for different levels: DEBUG, INFO, WARN, and ERROR.
 * Each log method accepts a formatted message and variadic arguments.
 *
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5, 2024
 */
class GPUZIPLogger {
public:
  // Log level constants
  static constexpr int DEBUG = 0;
  static constexpr int INFO = 1;
  static constexpr int WARN = 2;
  static constexpr int ERROR = 3;

  /**
   * @brief Sets the global logging level.
   *
   * @param level Log level to set (DEBUG, INFO, WARN, or ERROR).
   */
  static void SetLevel(int level) { sysLevel = level; }

  /**
   * @brief Enables or disables performance tracing for GPUZIP.
   *
   * This function allows toggling performance tracing on or off.
   * When enabled, GPUZIP logs performance-related data for debugging
   * and analysis.
   *
   * @param enable Set to true to enable performance tracing, false to disable
   * it.
   */
  static void PerfTraceSwitch(bool enable) { perfTrace = enable; }

  /**
   * @brief Enables or disables logging of compression metrics.
   *
   * This function controls whether compression metrics should be logged.
   * When enabled (`true`), calls to `LogCompressionMetrics` will output
   * compression details. When disabled (`false`), no logging will occur.
   *
   * @param enable Boolean flag to enable (`true`) or disable (`false`)
   *               compression metrics logging.
   */
  static void LogCompressionMetricsSwitch(bool enable) {
    logCompressionMetrics = enable;
  }

  /**
   * @brief Retrieves the current global logging level.
   *
   * @return The current log level.
   */
  static int GetLevel() { return sysLevel; }

  /**
   * @brief Checks if the current log level is DEBUG or higher.
   *
   * @return `true` if the logging level is set to DEBUG or a more verbose
   * level, otherwise `false`.
   */
  static bool IsDebug() { return sysLevel <= DEBUG; }

  /**
   * @brief Checks if the current log level is INFO or higher.
   *
   * @return `true` if the logging level is set to INFO, DEBUG, or a more
   * verbose level, otherwise `false`.
   */
  static bool IsInfo() { return sysLevel <= INFO; }

  /**
   * @brief Checks if the current log level is WARN or higher.
   *
   * @return `true` if the logging level is set to WARN, INFO, DEBUG, or a more
   * verbose level, otherwise `false`.
   */
  static bool IsWarn() { return sysLevel <= WARN; }

  /**
   * @brief Checks if the current log level is ERROR or higher.
   *
   * @return `true` if the logging level is set to ERROR, WARN, INFO, DEBUG, or
   * a more verbose level, otherwise `false`.
   */
  static bool IsError() { return sysLevel <= ERROR; }

  /**
   * @brief Checks if performance tracing logs should be recorded.
   *
   * @return `true` if performance tracing is enabled, otherwise `false`.
   */
  static bool ShouldLogPerfTrace() { return perfTrace; }

  /**
   * @brief Logs a performance tracing message if performance tracing is
   * enabled.
   *
   * @param message The format string (similar to `printf`).
   * @param ... Additional arguments for formatting the message.
   */
  static void PerfTrace(const char *message, ...) {
    if (ShouldLogPerfTrace()) {
      va_list args;
      va_start(args, message);
      fprintf(stderr, "[PERF] ");
      vfprintf(stderr, message, args);
      va_end(args);
    }
  }

  /**
   * @brief Logs a debug message if the log level is set to DEBUG or higher.
   *
   * @param message Format string for the debug message.
   * @param ... Variadic arguments for formatting.
   */
  static void Debug(const char *message, ...) {
    if (IsDebug()) {
      va_list args;
      va_start(args, message);
      fprintf(stderr, "[DEBUG] ");
      vfprintf(stderr, message, args);
      va_end(args);
    }
  }

  /**
   * @brief Logs an informational message if the log level is set to INFO or
   * higher.
   *
   * @param message Format string for the informational message.
   * @param ... Variadic arguments for formatting.
   */
  static void Info(const char *message, ...) {
    if (IsInfo()) {
      va_list args;
      va_start(args, message);
      fprintf(stderr, "[INFO] ");
      vfprintf(stderr, message, args);
      va_end(args);
    }
  }

  /**
   * @brief Logs a warning message if the log level is set to WARN or higher.
   *
   * @param message Format string for the warning message.
   * @param ... Variadic arguments for formatting.
   */
  static void Warn(const char *message, ...) {
    if (IsWarn()) {
      va_list args;
      va_start(args, message);
      fprintf(stderr, "[WARN] ");
      vfprintf(stderr, message, args);
      va_end(args);
    }
  }

  /**
   * @brief Logs an error message regardless of the current log level.
   *
   * @param message Format string for the error message.
   * @param ... Variadic arguments for formatting.
   */
  static void Error(const char *message, ...) {
    if (IsError()) {
      va_list args;
      va_start(args, message);
      fprintf(stderr, "[ERROR] ");
      vfprintf(stderr, message, args);
      va_end(args);
    }
  }

  /**
   * @brief Logs compression metrics for a given field at a specific timestep.
   *
   * This function calculates and logs the compression ratio (CR) of a dataset
   * stored in `field`. The compression ratio is computed as the ratio between
   * the uncompressed and compressed sizes. The log is printed to `stderr` in
   * CSV format.
   *
   * @param field Pointer to the `Field_t` structure containing data dimensions
   *              and compressed size.
   * @param timestep The timestep at which the compression occurred.
   * @param type An integer indicating the type of compression applied.
   */
  static void LogCompressionMetrics(Field_t *field, unsigned timestep,
                                    unsigned type) {
    if (logCompressionMetrics) {
      size_t uncompressed_len = field->n1 * field->n2 * field->n3 * sizeof(float);
      size_t compressed_len = field->size;
      double ratio = (double)uncompressed_len / (double)compressed_len;
      fprintf(stderr, "[CR], %i, %i, %lu, %lu, %.3f\n", timestep, type,
              uncompressed_len, field->size, ratio);
    }
  }

  /**
   * @brief log_prefetch prints information about the prefetch action vector.
   */
  static void LogPAV(PrefetchAction_t *pav) {
    fprintf(stderr, "Prefetch Action Vector\n");
    fprintf(stderr, "%-5s%-5s\n", "it", "ts");

    for (unsigned i = 0; i < pav->iter.size(); ++i) {
      fprintf(stderr, "%-5d", pav->iter[i]);
      fprintf(stderr, "%-5d", pav->timestep[i]);
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "=====================================\n");
  }

  /**
   * @brief log_pool prints information about the checkpoint pool.
   *
   * @param message A message to be printed before the table.
   */
  void LogPool(ChkptPool_t *pool, const char *message) {
    fprintf(stderr, "Checkpoint Pool (host): %s\n", message);
    fprintf(stderr, "%-6s%-10s%-10s%-8s\n", "Index", "Timestep", "Size",
            "Dims");

    for (unsigned i = 0; i < pool->size; ++i) {
      fprintf(stderr, "%-6d", i);
      fprintf(stderr, "%-10d", pool->timestep[i]);
      fprintf(stderr, "%-10zu", pool->currs[i].size);
      fprintf(stderr, "%zux%zux%zu", pool->currs[i].n1, pool->currs[i].n2,
              pool->currs[i].n3);
      fprintf(stderr, "\n");

      fprintf(stderr, "%-6d", i);
      fprintf(stderr, "%-10d", pool->timestep[i]);
      fprintf(stderr, "%-10zu", pool->prevs[i].size);
      fprintf(stderr, "%zux%zux%zu", pool->prevs[i].n1, pool->prevs[i].n2,
              pool->prevs[i].n3);
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "=====================================\n");
  }

  /**
   * @brief Logs the configuration settings for GPUZIP.
   *
   * This method prints the current configuration values using GPUZIPLogger,
   * providing a structured overview of checkpointing, compression, and logging
   * settings.
   *
   * @param gpuzip_config Pointer to the configuration structure.
   */
  static void LogConfig(const gpuzip_config_t *gpuzip_config) {
    fprintf(stderr, "GPUZIP Configuration:\n");

    /* 
    * Checkpointing algorithm settings
    */
    fprintf(stderr, "  Checkpointing Algorithm: %i (%s)\n",
            gpuzip_config->checkpointing_algorithm,
            gpuzip_config->checkpointing_algorithm == 0
                ? "TraceCheckpointing"
                : "RevolveCheckpointing");

    if (gpuzip_config->checkpointing_algorithm == 0) {
      fprintf(stderr, "   Trace File Path: %s\n",
              gpuzip_config->trace_file_path ? gpuzip_config->trace_file_path
                                             : "None");
    }

    if (gpuzip_config->checkpointing_algorithm == 1) {
      fprintf(stderr, "    Revolve Log Level: %i\n",
              gpuzip_config->revolve_log_level);
    }

    /* 
    * Prefetching settings
    */
    fprintf(stderr, "  Cache Capacity: %i\n", gpuzip_config->cache_capacity);

    /* 
    * Compressor settings
    */
    fprintf(stderr, "  Compressor: %i (%s)\n", gpuzip_config->compressor,
            gpuzip_config->compressor == 0   ? "None"
            : gpuzip_config->compressor == 1 ? "cuZFP"
            : gpuzip_config->compressor == 2 ? "Bitcomp"
                                             : "Unknown");

    if (gpuzip_config->compressor == 1) {
      fprintf(stderr, "    cuZFP Bit Rate: %i\n", gpuzip_config->zfp_bit_rate);
    }

    if (gpuzip_config->compressor == 2) {
      fprintf(stderr, "    Bitcomp Delta Config: %i (%s)\n",
              gpuzip_config->bitcomp_delta_config,
              gpuzip_config->bitcomp_delta_config == 0 ? "Max fraction strategy"
              : gpuzip_config->bitcomp_delta_config == 1 ? "Statistical range"
              : gpuzip_config->bitcomp_delta_config == 2 ? "Static delta"
                                                         : "Unknown");

      if (gpuzip_config->bitcomp_delta_config == 0) {
        fprintf(stderr, "    Bitcomp Range Fraction: %.6f\n",
                gpuzip_config->bitcomp_range_fraction);
      } else if (gpuzip_config->bitcomp_delta_config == 1) {
        fprintf(stderr, "    Bitcomp Num Sigma: %.6f\n",
                gpuzip_config->bitcomp_num_sigma);
      } else if (gpuzip_config->bitcomp_delta_config == 2) {
        fprintf(stderr, "    Bitcomp Delta: %e\n",
                gpuzip_config->bitcomp_delta);
      }

      fprintf(stderr, "    Bitcomp Algorithm: %i (%s)\n",
              gpuzip_config->bitcomp_algorithm,
              gpuzip_config->bitcomp_algorithm == 0 ? "Default" : "Sparse");
    }

    if (gpuzip_config->compressor == 3) {
      fprintf(stderr, "    cuSZp Error Bound: %e\n",
              gpuzip_config->cuszp_err_bound);
    }

    if (gpuzip_config->compressor > 0) {
      fprintf(stderr, "    Compression factor: %f\n",
              gpuzip_config->compression_factor);
    }

    fprintf(stderr, "  Log Level: %i (%s)\n", gpuzip_config->log_level,
            gpuzip_config->log_level == 0   ? "DEBUG"
            : gpuzip_config->log_level == 1 ? "INFO"
            : gpuzip_config->log_level == 2 ? "WARN"
                                            : "ERROR");

    fprintf(stderr, "  Enable Performance Log: %i\n",
            gpuzip_config->enable_performance_log);

    fprintf(stderr, "  Enable Compression Rate Log: %i\n",
              gpuzip_config->enable_compression_rate_log);
  }

private:
  // Global system log level
  static int sysLevel;

  static bool perfTrace;

  static bool logCompressionMetrics;
};

int GPUZIPLogger::sysLevel = GPUZIPLogger::DEBUG;
bool GPUZIPLogger::perfTrace = false;
bool GPUZIPLogger::logCompressionMetrics = false;
