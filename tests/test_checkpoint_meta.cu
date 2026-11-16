/**
 * GPU tests for the checkpoint metadata of Save/Retrieve (CheckpointMeta_t).
 *
 * The checkpoint metadata is opaque bytes supplied with each Save and returned
 * by the Retrieve of the same timestep, whatever path the fields took (cache
 * hit, prefetch, host pool). These tests run full checkpointing cycles and check
 * on every restore that the fields and the metadata are those of that timestep:
 *
 *   meta_prefetch_bitcomp    Revolve, prefetching cache of 2, Bitcomp.
 *   meta_prefetch_raw        Revolve, prefetching cache of 2, no compressor.
 *   meta_chkptonly_bitcomp   Revolve, CheckpointOnly (cache 0), Bitcomp.
 *   meta_chkptonly_raw       Revolve, CheckpointOnly (cache 0), no compressor.
 *   meta_uniform_bitcomp     Uniform, prefetching cache of 2, Bitcomp.
 *
 * One restore in three passes no metadata (allowed: it is ignored). The
 * prefetching cache never misses in these cycles: its planner finds room for a
 * prefetch of every restore (checked for Revolve with 10 to 500 steps and
 * caches of 2 to 4, and for Uniform). The miss path of Prefetch, where the
 * fields and the metadata come from the host pool, is forced by saving three
 * checkpoints into a cache of 2 without Dispatch and retrieving them in reverse
 * order: the first one saved has left the cache.
 *
 *   meta_miss_raw            Forced cache miss, no compressor.
 *   meta_miss_bitcomp        Forced cache miss, Bitcomp.
 *
 * The misuse
 * cases must stop the run with a "GPUZIP error" message (checked by CTest):
 *
 *   meta_absent_rejected     Retrieve with metadata of a checkpoint saved without it.
 *   meta_len_mismatch        Two Saves with metadata of different lengths.
 *   meta_buffer_mismatch     Retrieve into a buffer of another length.
 *
 * Requires an NVIDIA GPU. Usage: test_checkpoint_meta <test name>
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/GPUZIPBuilders.cpp"
#include "common/GPUZIPConfig.h"
#include "CompressorBitcomp.hpp"

#include "test_common.cuh"

static const int N1 = 64, N2 = 64, N3 = 64, STEPS = 60;
static const float NOISE = 0.05f;
static const double DELTA = 1e-8;
static const size_t META_LEN = 4680; // checkpoint metadata of a large-v2 segment, 4 GPUs

/// The metadata saved with timestep `ts`: different for every timestep.
static void fill_meta(std::vector<uint8_t> &m, int ts) {
  for (size_t i = 0; i < m.size(); i++) m[i] = (uint8_t)(ts * 131 + i * 7 + (i >> 8));
}

static gpuzip_config_t make_config(int algorithm, int cache_capacity, int compressor) {
  gpuzip_config_t c;
  memset(&c, 0, sizeof(c));
  c.checkpointing_algorithm = algorithm; // 1 Revolve, 2 Uniform
  c.checkpointing_snaps = 6;             // Uniform only
  c.cache_capacity = cache_capacity;
  c.compressor = compressor;             // 0 none, 2 Bitcomp
  c.log_level = 2;                       // WARN
  c.compression_factor = 0;              // size the slots by the compressor's worst case
  c.float_kind = "float";
  c.bitcomp_delta_config = 2;            // static delta
  c.bitcomp_delta = DELTA;
  c.bitcomp_algorithm = 0;
  return c;
}

/// A full checkpointing cycle with metadata. Returns the number of failed checks.
static int run_meta_cycle(int algorithm, int cache_capacity, int compressor) {
  const size_t n = (size_t)N1 * N2 * N3, raw = n * sizeof(float);
  gpuzip_config_t cfg = make_config(algorithm, cache_capacity, compressor);
  GPUZIPLogger::SetLevel(cfg.log_level);
  const bool compressed = compressor > 0;

  float *d_curr, *d_prev, *d_ref;
  CHECK_CUDA(cudaMalloc(&d_curr, raw));
  CHECK_CUDA(cudaMalloc(&d_prev, raw));
  CHECK_CUDA(cudaMalloc(&d_ref, raw));
  Field_t curr = {d_curr, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  Field_t prev = {d_prev, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  Checkpointing *chkpt = GPUZIPBuilders::CheckpointingBuilder(&cfg, STEPS);
  Prefetch *prefetch = GPUZIPBuilders::PrefetchBuilder(&cfg, N1, N2, N3, STEPS, chkpt);
  std::unique_ptr<Compressor<void, void>> comp_curr, comp_prev;
  if (compressed) {
    comp_curr = GPUZIPBuilders::CompressorBuilder(&cfg, N1, N2, N3);
    comp_prev = GPUZIPBuilders::CompressorBuilder(&cfg, N1, N2, N3);
  }
  prefetch->Setup();

  std::vector<uint8_t> saved(META_LEN), got(META_LEN), expected(META_LEN);
  int bad = 0, saves = 0, restores = 0, with_meta = 0;
  make_field(d_curr, N1, N2, N3, 0, 0, NOISE);
  make_field(d_prev, N1, N2, N3, 0, 1, NOISE);
  chkpt->Reset();
  Action a = chkpt->GetAction();
  while (a.actionType != ACTION_TERMINATE && a.actionType != ACTION_ERROR) {
    prefetch->Dispatch(chkpt->GetIt());
    if (a.actionType == ACTION_FORWARD) {
      make_field(d_curr, N1, N2, N3, a.ts, 0, NOISE);
      make_field(d_prev, N1, N2, N3, a.ts, 1, NOISE);
    } else if (a.actionType == ACTION_SAVE) {
      curr.size = raw; prev.size = raw;
      fill_meta(saved, a.ts);
      CheckpointMeta_t m = {saved.data(), META_LEN};
      if (compressed)
        prefetch->Save((unsigned)a.ts, &curr, &prev, comp_curr.get(), comp_prev.get(), &m);
      else
        prefetch->Save(a.ts, &curr, &prev, &m);
      std::fill(saved.begin(), saved.end(), 0x5A); // GPUZIP must have kept a copy
      saves++;
    } else if (a.actionType == ACTION_RESTORE) {
      restores++;
      const bool use_meta = restores % 3 != 0;
      std::fill(got.begin(), got.end(), 0xEE);
      CheckpointMeta_t g = {got.data(), META_LEN};
      CheckpointMeta_t *gp = use_meta ? &g : nullptr;
      if (compressed)
        prefetch->Retrieve((unsigned)a.ts, &curr, &prev, comp_curr.get(), comp_prev.get(), gp);
      else
        prefetch->Retrieve(a.ts, &curr, &prev, stream, gp);
      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaDeviceSynchronize());
      curr.size = raw; prev.size = raw;
      for (int which = 0; which < 2; which++) {
        make_field(d_ref, N1, N2, N3, a.ts, which, NOISE);
        float err = max_abs_diff(which == 0 ? d_curr : d_prev, d_ref, n);
        if (err > (compressed ? DELTA : 0.0)) {
          bad++;
          printf("restore ts=%d %s: max error %.3e\n", a.ts, which == 0 ? "curr" : "prev", err);
        }
      }
      if (use_meta) {
        with_meta++;
        fill_meta(expected, a.ts);
        if (got != expected) {
          bad++;
          size_t i = 0;
          while (got[i] == expected[i]) i++;
          printf("restore ts=%d: metadata differs from byte %zu (got %u, expected %u)\n", a.ts,
                 i, got[i], expected[i]);
        }
      } else if (got != std::vector<uint8_t>(META_LEN, 0xEE)) {
        bad++;
        printf("restore ts=%d without metadata: the buffer was written\n", a.ts);
      }
    }
    a = chkpt->GetAction();
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  if (a.actionType == ACTION_ERROR) {
    printf("checkpointing returned ACTION_ERROR\n");
    bad++;
  }
  const int hits = prefetch->GetHitsCount(), prefetched = prefetch->GetPrefetchCount(),
            misses = prefetch->GetMissesCount();
  printf("%d saves, %d restores (%d with metadata): %d cache hits (%d prefetched), %d misses\n",
         saves, restores, with_meta, hits, prefetched, misses);
  // The cases must exercise the paths they are meant for.
  if (restores == 0 || with_meta == 0) bad++;
  if (cache_capacity >= 2 && (hits == 0 || prefetched == 0)) {
    printf("the prefetching cache was not exercised\n");
    bad++;
  }
  prefetch->Report();
  prefetch->Free();
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_curr));
  CHECK_CUDA(cudaFree(d_prev));
  CHECK_CUDA(cudaFree(d_ref));
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

/// Forced cache miss of Prefetch: Save ts 0, 1, 2 into a cache of 2 without
/// Dispatch, then Retrieve 2, 1 (hits) and 0 (miss). Returns the number of failed checks.
static int run_forced_miss(int compressor) {
  const int SAVES = 3, CACHE = 2;
  const size_t n = (size_t)N1 * N2 * N3, raw = n * sizeof(float);
  gpuzip_config_t cfg = make_config(1, CACHE, compressor);
  GPUZIPLogger::SetLevel(cfg.log_level);
  const bool compressed = compressor > 0;

  float *d_curr, *d_prev, *d_ref;
  CHECK_CUDA(cudaMalloc(&d_curr, raw));
  CHECK_CUDA(cudaMalloc(&d_prev, raw));
  CHECK_CUDA(cudaMalloc(&d_ref, raw));
  Field_t curr = {d_curr, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  Field_t prev = {d_prev, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  Checkpointing *chkpt = GPUZIPBuilders::CheckpointingBuilder(&cfg, STEPS);
  if (chkpt->GetNumberOfCheckpoints() < SAVES) {
    printf("FAIL: the pool has %d slots, %d needed\n", chkpt->GetNumberOfCheckpoints(), SAVES);
    return 1;
  }
  Prefetch *prefetch = GPUZIPBuilders::PrefetchBuilder(&cfg, N1, N2, N3, STEPS, chkpt);
  std::unique_ptr<Compressor<void, void>> comp_curr, comp_prev;
  if (compressed) {
    comp_curr = GPUZIPBuilders::CompressorBuilder(&cfg, N1, N2, N3);
    comp_prev = GPUZIPBuilders::CompressorBuilder(&cfg, N1, N2, N3);
  }
  prefetch->Setup();

  std::vector<uint8_t> saved(META_LEN), got(META_LEN), expected(META_LEN);
  int bad = 0;
  for (int ts = 0; ts < SAVES; ts++) {
    make_field(d_curr, N1, N2, N3, ts, 0, NOISE);
    make_field(d_prev, N1, N2, N3, ts, 1, NOISE);
    curr.size = raw; prev.size = raw;
    fill_meta(saved, ts);
    CheckpointMeta_t m = {saved.data(), META_LEN};
    if (compressed)
      prefetch->Save((unsigned)ts, &curr, &prev, comp_curr.get(), comp_prev.get(), &m);
    else
      prefetch->Save(ts, &curr, &prev, &m);
    std::fill(saved.begin(), saved.end(), 0x5A);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  for (int ts = SAVES - 1; ts >= 0; ts--) {
    CHECK_CUDA(cudaMemset(d_curr, 0, raw));
    CHECK_CUDA(cudaMemset(d_prev, 0, raw));
    std::fill(got.begin(), got.end(), 0xEE);
    CheckpointMeta_t g = {got.data(), META_LEN};
    if (compressed)
      prefetch->Retrieve((unsigned)ts, &curr, &prev, comp_curr.get(), comp_prev.get(), &g);
    else
      prefetch->Retrieve(ts, &curr, &prev, stream, &g);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaDeviceSynchronize());
    curr.size = raw; prev.size = raw;
    for (int which = 0; which < 2; which++) {
      make_field(d_ref, N1, N2, N3, ts, which, NOISE);
      float err = max_abs_diff(which == 0 ? d_curr : d_prev, d_ref, n);
      if (err > (compressed ? DELTA : 0.0)) {
        bad++;
        printf("restore ts=%d %s: max error %.3e\n", ts, which == 0 ? "curr" : "prev", err);
      }
    }
    fill_meta(expected, ts);
    if (got != expected) {
      bad++;
      printf("restore ts=%d: the metadata differs\n", ts);
    }
  }
  const int hits = prefetch->GetHitsCount(), misses = prefetch->GetMissesCount();
  printf("%d saves, %d restores: %d cache hits, %d misses\n", SAVES, SAVES, hits, misses);
  if (hits != SAVES - 1 || misses != 1) {
    printf("expected %d hits and 1 miss\n", SAVES - 1);
    bad++;
  }
  prefetch->Free();
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_curr));
  CHECK_CUDA(cudaFree(d_prev));
  CHECK_CUDA(cudaFree(d_ref));
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

/// Misuse cases: CheckpointOnly without compressor, two checkpoints saved directly.
static int run_misuse(const std::string &test) {
  const size_t raw = (size_t)N1 * N2 * N3 * sizeof(float);
  gpuzip_config_t cfg = make_config(1, 0, 0);
  GPUZIPLogger::SetLevel(cfg.log_level);
  float *d_curr, *d_prev;
  CHECK_CUDA(cudaMalloc(&d_curr, raw));
  CHECK_CUDA(cudaMalloc(&d_prev, raw));
  CHECK_CUDA(cudaMemset(d_curr, 0, raw));
  CHECK_CUDA(cudaMemset(d_prev, 0, raw));
  Field_t curr = {d_curr, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  Field_t prev = {d_prev, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  Checkpointing *chkpt = GPUZIPBuilders::CheckpointingBuilder(&cfg, STEPS);
  Prefetch *prefetch = GPUZIPBuilders::PrefetchBuilder(&cfg, N1, N2, N3, STEPS, chkpt);
  prefetch->Setup();

  std::vector<uint8_t> buf(META_LEN + 1, 1);
  CheckpointMeta_t m = {buf.data(), META_LEN}, longer = {buf.data(), META_LEN + 1};
  if (test == "meta_absent_rejected") {
    prefetch->Save(0, &curr, &prev);
    prefetch->Retrieve(0, &curr, &prev, (cudaStream_t)0, &m);
  } else if (test == "meta_len_mismatch") {
    prefetch->Save(0, &curr, &prev, &m);
    prefetch->Save(1, &curr, &prev, &longer);
  } else if (test == "meta_buffer_mismatch") {
    prefetch->Save(0, &curr, &prev, &m);
    prefetch->Retrieve(0, &curr, &prev, (cudaStream_t)0, &longer);
  }
  // Expected to stop inside GPUZIP with a "GPUZIP error" message (CTest checks
  // the output). Reaching this point means the misuse went unnoticed.
  printf("FAIL: the run completed although the checkpoint metadata was misused\n");
  return 1;
}

int main(int argc, char **argv) {
  const std::string test = argc > 1 ? argv[1] : "";
  int dev = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("GPU %d: %s\n", dev, prop.name);
  if (test == "meta_prefetch_bitcomp") return run_meta_cycle(1, 2, 2);
  if (test == "meta_prefetch_raw") return run_meta_cycle(1, 2, 0);
  if (test == "meta_chkptonly_bitcomp") return run_meta_cycle(1, 0, 2);
  if (test == "meta_chkptonly_raw") return run_meta_cycle(1, 0, 0);
  if (test == "meta_uniform_bitcomp") return run_meta_cycle(2, 2, 2);
  if (test == "meta_miss_raw") return run_forced_miss(0);
  if (test == "meta_miss_bitcomp") return run_forced_miss(2);
  if (test == "meta_absent_rejected" || test == "meta_len_mismatch" ||
      test == "meta_buffer_mismatch")
    return run_misuse(test);
  fprintf(stderr, "usage: %s <test name> (see the comment at the top)\n", argv[0]);
  return 2;
}
