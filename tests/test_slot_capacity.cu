/**
 * GPU tests for the checkpoint slot capacity (LSC-Unicamp/GPUZIP#24).
 *
 * Pool and cache slots are sized as field size / CompressionFactor (rounded up
 * to cudaMalloc's 2 MiB granularity, see GPUZIPSlotCapacity), and the
 * compressors take no output capacity. These tests check the assumptions the
 * size checks rely on and the behaviour of Prefetch around the slot limit:
 *
 *   bitcomp_write_extent   Bitcomp writes nothing past the compressed size it
 *                          reports (guard bytes after it stay untouched), and
 *                          decompression stays within the error bound.
 *   slot_fits_rounding     A field larger than the nominal slot but within the
 *                          rounded capacity is saved and restored correctly
 *                          through Prefetch (Revolve + prefetching cache).
 *   slot_overflow_detected A field larger than the slot capacity stops the run
 *                          with a "GPUZIP error" message (checked by CTest).
 *
 * Requires an NVIDIA GPU. Usage: test_slot_capacity <test name>
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/GPUZIPBuilders.cpp"
#include "common/GPUZIPConfig.h"
#include "CompressorBitcomp.hpp"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                             \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

static const unsigned char GUARD_BYTE = 0xA5;
static const size_t GUARD_LEN = (size_t)1 << 20;

/// Deterministic pseudo-random value in [-1, 1).
__device__ float hash01(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return (float)(x & 0xffffff) / (float)0x800000 - 1.0f;
}

/// A wave-like field: a smooth Gaussian-windowed plane wave that moves with
/// `ts`, plus white noise of amplitude `noise` (which sets the compressibility).
__global__ void fill_field(float *f, int n1, int n2, int n3, int ts, int which,
                           float noise) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t total = (size_t)n1 * n2 * n3;
  if (idx >= total) return;
  int i = idx % n1, j = (idx / n1) % n2, k = idx / ((size_t)n1 * n2);
  float x = (i - 0.5f * n1) / n1, y = (j - 0.5f * n2) / n2, z = (k - 0.5f * n3) / n3;
  float r2 = x * x + y * y + z * z;
  float smooth = expf(-8.0f * r2) * sinf(40.0f * sqrtf(r2) - 0.2f * ts - 0.5f * which);
  f[idx] = smooth + noise * hash01((uint32_t)(idx * 2654435761u) ^ (uint32_t)(ts * 97 + which));
}

static void make_field(float *d, int n1, int n2, int n3, int ts, int which, float noise) {
  size_t total = (size_t)n1 * n2 * n3;
  fill_field<<<(unsigned)((total + 255) / 256), 256>>>(d, n1, n2, n3, ts, which, noise);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

static float max_abs_diff(const float *d_a, const float *d_b, size_t n) {
  std::vector<float> a(n), b(n);
  CHECK_CUDA(cudaMemcpy(a.data(), d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
  float m = 0.0f;
  for (size_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

// ---------------------------------------------------------------------------
// bitcomp_write_extent
// ---------------------------------------------------------------------------
static int test_bitcomp_write_extent() {
  struct Shape { int n1, n2, n3; };
  const Shape shapes[] = {{128, 128, 128}, {101, 77, 53}};
  const float noises[] = {0.0f, 1e-6f, 1e-4f, 1e-2f, 1.0f};
  const double deltas[] = {1e-8, 1e-4, 1e-2};
  const char *algos[] = {"default", "sparse"};
  int failures = 0, cases = 0;

  for (const Shape &s : shapes) {
    const size_t n = (size_t)s.n1 * s.n2 * s.n3, raw = n * sizeof(float);
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, raw));
    CHECK_CUDA(cudaMalloc(&d_out, raw));
    for (const char *algo : algos) {
      for (double delta : deltas) {
        CompressorBitcomp<void, void> comp(s.n1, s.n2, s.n3, 2, 0.0, 0.0, delta,
                                           "float", algo);
        const size_t buf_len = comp.CompressedMaxSize() + GUARD_LEN;
        unsigned char *d_buf;
        CHECK_CUDA(cudaMalloc(&d_buf, buf_len));
        std::vector<unsigned char> h_buf(buf_len);

        for (int which = -1; which < (int)(sizeof(noises) / sizeof(noises[0])); which++) {
          // which = -1: all zeros
          if (which < 0) CHECK_CUDA(cudaMemset(d_in, 0, raw));
          else make_field(d_in, s.n1, s.n2, s.n3, 7, 0, noises[which]);

          CHECK_CUDA(cudaMemset(d_buf, GUARD_BYTE, buf_len));
          const size_t size = comp.Compress(d_in, d_buf);
          CHECK_CUDA(cudaDeviceSynchronize());
          CHECK_CUDA(cudaMemcpy(h_buf.data(), d_buf, buf_len, cudaMemcpyDeviceToHost));

          size_t touched = 0, last = 0;
          for (size_t b = size; b < buf_len; b++) {
            if (h_buf[b] != GUARD_BYTE) { touched++; last = b; }
          }
          comp.Decompress(d_buf, d_out);
          CHECK_CUDA(cudaDeviceSynchronize());
          const float err = max_abs_diff(d_in, d_out, n);
          const bool ok = touched == 0 && size <= comp.CompressedMaxSize() && err <= delta;
          cases++;
          if (!ok) failures++;
          char field[32];
          if (which < 0) snprintf(field, sizeof(field), "zeros");
          else snprintf(field, sizeof(field), "noise=%g", noises[which]);
          printf("%s %dx%dx%d algo=%-7s delta=%.0e field=%-11s ratio=%9.3f size=%zu "
                 "bytes_after_size_modified=%zu",
                 ok ? "ok  " : "FAIL", s.n1, s.n2, s.n3, algo, delta, field,
                 (double)raw / size, size, touched);
          if (touched) printf(" (last at size+%zu)", last - size);
          printf(" max_err=%.3e\n", err);
        }
        CHECK_CUDA(cudaFree(d_buf));
      }
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
  }
  printf("%s: %d of %d cases failed\n", failures ? "FAIL" : "PASS", failures, cases);
  return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Checkpoint cycle through Prefetch (Revolve, prefetching cache of 2)
// ---------------------------------------------------------------------------
static const int N1 = 128, N2 = 128, N3 = 128, STEPS = 40;
static const float NOISE = 0.05f;   // compresses ~1.25x with delta=1e-8
static const double DELTA = 1e-8;

static gpuzip_config_t make_config(float compression_factor) {
  gpuzip_config_t c;
  memset(&c, 0, sizeof(c));
  c.checkpointing_algorithm = 1;   // Revolve
  c.cache_capacity = 2;
  c.compressor = 2;                // Bitcomp
  c.log_level = 2;                 // WARN
  c.compression_factor = compression_factor;
  c.float_kind = "float";
  c.bitcomp_delta_config = 2;      // static delta
  c.bitcomp_delta = DELTA;
  c.bitcomp_algorithm = 0;
  return c;
}

/// Compressed sizes of all fields the cycle will save: {min, max}.
static void probe_sizes(size_t &cmin, size_t &cmax) {
  const size_t raw = (size_t)N1 * N2 * N3 * sizeof(float);
  float *d_f; void *d_buf;
  CompressorBitcomp<void, void> comp(N1, N2, N3, 2, 0.0, 0.0, DELTA, "float", "default");
  CHECK_CUDA(cudaMalloc(&d_f, raw));
  CHECK_CUDA(cudaMalloc(&d_buf, comp.CompressedMaxSize()));
  cmin = SIZE_MAX; cmax = 0;
  for (int ts = 0; ts <= STEPS; ts++) {
    for (int which = 0; which < 2; which++) {
      make_field(d_f, N1, N2, N3, ts, which, NOISE);
      size_t s = comp.Compress(d_f, d_buf);
      cmin = s < cmin ? s : cmin; cmax = s > cmax ? s : cmax;
    }
  }
  CHECK_CUDA(cudaFree(d_f));
  CHECK_CUDA(cudaFree(d_buf));
}

/// Runs a full Revolve cycle; on every restore compares the fields with the
/// ones that were saved. Returns the number of restores outside the bound.
static int run_cycle(float compression_factor, int &restores) {
  const size_t n = (size_t)N1 * N2 * N3, raw = n * sizeof(float);
  gpuzip_config_t cfg = make_config(compression_factor);
  GPUZIPLogger::SetLevel(cfg.log_level);

  float *d_curr, *d_prev, *d_ref;
  CHECK_CUDA(cudaMalloc(&d_curr, raw));
  CHECK_CUDA(cudaMalloc(&d_prev, raw));
  CHECK_CUDA(cudaMalloc(&d_ref, raw));
  Field_t curr = {d_curr, raw, (size_t)N1, (size_t)N2, (size_t)N3};
  Field_t prev = {d_prev, raw, (size_t)N1, (size_t)N2, (size_t)N3};

  Checkpointing *chkpt = GPUZIPBuilders::CheckpointingBuilder(&cfg, STEPS);
  Prefetch *prefetch = GPUZIPBuilders::PrefetchBuilder(&cfg, N1, N2, N3, STEPS, chkpt);
  auto comp_curr = GPUZIPBuilders::CompressorBuilder(&cfg, N1, N2, N3);
  auto comp_prev = GPUZIPBuilders::CompressorBuilder(&cfg, N1, N2, N3);
  prefetch->Setup();

  int bad = 0;
  restores = 0;
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
      prefetch->Save((unsigned)a.ts, &curr, &prev, comp_curr.get(), comp_prev.get());
    } else if (a.actionType == ACTION_RESTORE) {
      prefetch->Retrieve((unsigned)a.ts, &curr, &prev, comp_curr.get(), comp_prev.get());
      CHECK_CUDA(cudaDeviceSynchronize());
      curr.size = raw; prev.size = raw;
      restores++;
      for (int which = 0; which < 2; which++) {
        make_field(d_ref, N1, N2, N3, a.ts, which, NOISE);
        float err = max_abs_diff(which == 0 ? d_curr : d_prev, d_ref, n);
        if (err > DELTA) {
          bad++;
          printf("restore ts=%d %s: max error %.3e > %.0e\n", a.ts,
                 which == 0 ? "curr" : "prev", err, DELTA);
        }
      }
    }
    a = chkpt->GetAction();
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  if (a.actionType == ACTION_ERROR) {
    printf("checkpointing returned ACTION_ERROR\n");
    bad++;
  }
  prefetch->Report();
  prefetch->Free();
  CHECK_CUDA(cudaFree(d_curr));
  CHECK_CUDA(cudaFree(d_prev));
  CHECK_CUDA(cudaFree(d_ref));
  return bad;
}

static const size_t MIB2 = (size_t)2 << 20;

static int test_slot_fits_rounding() {
  const size_t raw = (size_t)N1 * N2 * N3 * sizeof(float);
  size_t cmin, cmax;
  probe_sizes(cmin, cmax);
  // Nominal slot strictly between the 2 MiB step below cmax and cmax itself:
  // the largest fields exceed it, but its rounded capacity still holds them.
  const size_t step_below = ((cmax - 1) / MIB2) * MIB2;
  const size_t nominal = step_below + (cmax - step_below) / 2;
  const float cf = (float)((double)raw / nominal);
  printf("raw %zu B, compressed %zu..%zu B, CompressionFactor %.6f -> nominal slot "
         "%zu B, capacity %zu B\n", raw, cmin, cmax, cf,
         (size_t)((double)raw / cf), GPUZIPSlotCapacity((size_t)((double)raw / cf)));
  if (!(cmax > (size_t)((double)raw / cf) && cmax <= GPUZIPSlotCapacity((size_t)((double)raw / cf)))) {
    printf("FAIL: could not set up a field between the nominal slot and its capacity\n");
    return 1;
  }
  int restores = 0;
  int bad = run_cycle(cf, restores);
  printf("%s: %d restores, %d outside the error bound\n", bad ? "FAIL" : "PASS",
         restores, bad);
  return bad || restores == 0 ? 1 : 0;
}

static int test_slot_overflow_detected() {
  const size_t raw = (size_t)N1 * N2 * N3 * sizeof(float);
  size_t cmin, cmax;
  probe_sizes(cmin, cmax);
  // Capacity well below the smallest compressed field.
  const size_t nominal = cmin / 2;
  const float cf = (float)((double)raw / nominal);
  printf("raw %zu B, compressed %zu..%zu B, CompressionFactor %.3f -> capacity %zu B\n",
         raw, cmin, cmax, cf, GPUZIPSlotCapacity((size_t)((double)raw / cf)));
  if (GPUZIPSlotCapacity((size_t)((double)raw / cf)) >= cmin) {
    printf("FAIL: setup, capacity not below the compressed size\n");
    return 1;
  }
  int restores = 0;
  run_cycle(cf, restores);
  // Expected to stop inside Save() with a "GPUZIP error" message (CTest checks
  // the output). Reaching this point means the overflow went unnoticed.
  printf("FAIL: the run completed although no field fits its slot\n");
  return 1;
}

int main(int argc, char **argv) {
  const std::string test = argc > 1 ? argv[1] : "";
  int dev = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("GPU %d: %s\n", dev, prop.name);
  if (test == "bitcomp_write_extent") return test_bitcomp_write_extent();
  if (test == "slot_fits_rounding") return test_slot_fits_rounding();
  if (test == "slot_overflow_detected") return test_slot_overflow_detected();
  fprintf(stderr,
          "usage: %s bitcomp_write_extent|slot_fits_rounding|slot_overflow_detected\n",
          argv[0]);
  return 2;
}
