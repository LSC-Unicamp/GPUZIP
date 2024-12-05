#pragma once

typedef struct {
  // TraceCheckpointing=0 or RevolveCheckpointing=1
  int checkpointing_algorithm;

  // 0 no prefetch. For prefetching, the capacity needs at minimum 2.
  int cache_capacity;

  // Empty=0 (for no compression), cuZFP=1 or Bitcomp=2. Integer.
  int compressor;

  // 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
  int log_level;

  // Useful for logging performance insights
  bool enable_performance_log;

  // Logs compression rate  -- VERY verbose
  bool enable_compression_rate_log;

  /// Error bound compressors are not good on predict its compression ration
  /// Knowing your model, you can set the worse compression ratio factor in other
  /// to make more checkpointing. 
  float compression_factor;

  // path to the Trace file
  char *trace_file_path;

  // Revolve's checkpointing log level (0 to 4)
  int revolve_log_level;

  // cuZFP's max bits.
  // https://zfp.readthedocs.io/en/release0.5.4/modes.html#mode-fixed-rate
  int zfp_bit_rate;

  // cuSZp's error bound parameter. Double.
  double cuszp_err_bound;

  // The way Bitcomp's delta parameter will be set. Integer.
  // 0 = Max fraction strategy -- Used max value in data scaled by `MaxFraction`
  // 1 = Statistical range -- Used mean +- `NumSigma` * stddev for scaling
  // 2 = Static delta -- considers a fixed `Delta` value
  int bitcomp_delta_config;

  // Case BitcompDeltaConfig=0. Double
  double bitcomp_range_fraction;

  // Case BitcompDeltaConfig=1. Double.
  double bitcomp_num_sigma;

  // Case BitcompDeltaConfig=2. Float.
  double bitcomp_delta;

  // Bitcomp's algorithm (`0` for default or `1` for sparse). Integer.
  int bitcomp_algorithm;
} gpuzip_config_t;
