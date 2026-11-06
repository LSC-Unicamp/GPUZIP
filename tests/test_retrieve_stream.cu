/**
 * GPU tests for the ordering of Retrieve without compressor against the
 * caller's stream (PrivateGPUZIP issue #3).
 *
 * A solver can still have work queued on its stream that uses the fields when
 * it calls Retrieve(timestep, curr, prev, stream): awave3d's imaging condition
 * reads the forward field just before a restore. The copies into curr and prev
 * must start only after that work. Each restore here first queues, on the
 * caller's stream, a kernel that waits WAIT_MS and then copies curr; that copy
 * must still see the field from before the restore (zeros), and after the
 * Retrieve curr and prev must hold the checkpoint.
 *
 *   order_prefetch_raw     Prefetching cache of 2: Save ts 0, 1, 2 without
 *                          Dispatch and Retrieve 2, 1 (hits) and 0 (miss).
 *   order_chkptonly_raw    CheckpointOnly (cache 0), same saves and restores.
 *
 * Requires an NVIDIA GPU. Usage: test_retrieve_stream <test name>
 */

#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/GPUZIPBuilders.cpp"
#include "common/GPUZIPConfig.h"

#include "test_common.cuh"

static const int N1 = 64, N2 = 64, N3 = 64, STEPS = 60, SAVES = 3;
static const float NOISE = 0.05f;
static const unsigned long long WAIT_MS = 100;

/// Waits `ns` nanoseconds (GPU global timer), then copies `n` floats of src to dst.
__global__ void wait_then_copy(const float *src, float *dst, size_t n,
                               unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < ns);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

static gpuzip_config_t make_config(int cache_capacity) {
  gpuzip_config_t c;
  memset(&c, 0, sizeof(c));
  c.checkpointing_algorithm = 1; // Revolve
  c.cache_capacity = cache_capacity;
  c.compressor = 0;
  c.log_level = 2; // WARN
  c.float_kind = "float";
  return c;
}

/// Returns the number of failed checks.
static int run_order(int cache_capacity) {
  const size_t n = (size_t)N1 * N2 * N3, raw = n * sizeof(float);
  gpuzip_config_t cfg = make_config(cache_capacity);
  GPUZIPLogger::SetLevel(cfg.log_level);

  float *d_curr, *d_prev, *d_ref, *d_seen, *d_zero;
  CHECK_CUDA(cudaMalloc(&d_curr, raw));
  CHECK_CUDA(cudaMalloc(&d_prev, raw));
  CHECK_CUDA(cudaMalloc(&d_ref, raw));
  CHECK_CUDA(cudaMalloc(&d_seen, raw));
  CHECK_CUDA(cudaMalloc(&d_zero, raw));
  CHECK_CUDA(cudaMemset(d_zero, 0, raw));
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
  prefetch->Setup();

  for (int ts = 0; ts < SAVES; ts++) {
    make_field(d_curr, N1, N2, N3, ts, 0, NOISE);
    make_field(d_prev, N1, N2, N3, ts, 1, NOISE);
    curr.size = raw; prev.size = raw;
    prefetch->Save(ts, &curr, &prev);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  int bad = 0;
  for (int ts = SAVES - 1; ts >= 0; ts--) {
    CHECK_CUDA(cudaMemset(d_curr, 0, raw));
    CHECK_CUDA(cudaMemset(d_prev, 0, raw));
    CHECK_CUDA(cudaMemset(d_seen, 0x3f, raw)); // 0.747 if the kernel never copies
    CHECK_CUDA(cudaDeviceSynchronize());
    wait_then_copy<<<132, 256, 0, stream>>>(d_curr, d_seen, n, WAIT_MS * 1000000ULL);
    CHECK_CUDA(cudaGetLastError());
    prefetch->Retrieve(ts, &curr, &prev, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaDeviceSynchronize());
    curr.size = raw; prev.size = raw;

    float seen = max_abs_diff(d_seen, d_zero, n);
    if (seen != 0.0f) {
      bad++;
      printf("restore ts=%d: curr was overwritten before the work queued on the stream "
             "(it saw max |value| %.3e instead of 0)\n", ts, seen);
    }
    for (int which = 0; which < 2; which++) {
      make_field(d_ref, N1, N2, N3, ts, which, NOISE);
      float err = max_abs_diff(which == 0 ? d_curr : d_prev, d_ref, n);
      if (err != 0.0f) {
        bad++;
        printf("restore ts=%d %s: max error %.3e\n", ts, which == 0 ? "curr" : "prev", err);
      }
    }
  }
  const int hits = prefetch->GetHitsCount(), misses = prefetch->GetMissesCount();
  printf("%d saves, %d restores: %d cache hits, %d misses\n", SAVES, SAVES, hits, misses);
  // The prefetching case must exercise both paths of Prefetch::retrieve.
  if (cache_capacity >= 2 && (hits != SAVES - 1 || misses != 1)) {
    printf("expected %d hits and 1 miss\n", SAVES - 1);
    bad++;
  }
  prefetch->Free();
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_curr));
  CHECK_CUDA(cudaFree(d_prev));
  CHECK_CUDA(cudaFree(d_ref));
  CHECK_CUDA(cudaFree(d_seen));
  CHECK_CUDA(cudaFree(d_zero));
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

int main(int argc, char **argv) {
  const std::string test = argc > 1 ? argv[1] : "";
  int dev = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("GPU %d: %s\n", dev, prop.name);
  if (test == "order_prefetch_raw") return run_order(2);
  if (test == "order_chkptonly_raw") return run_order(0);
  fprintf(stderr, "usage: %s <test name> (see the comment at the top)\n", argv[0]);
  return 2;
}
