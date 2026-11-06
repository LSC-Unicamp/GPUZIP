# GPUZIP tests

GPU tests (they need an NVIDIA GPU to run). A first step towards #9.

| Test | What it checks |
|------|----------------|
| `bitcomp_write_extent` | Bitcomp writes nothing past the compressed size it reports: the buffer is filled with a guard byte before compressing and every byte after that size must be unchanged. Also checks the decompression error bound. Covers zeros, smooth and noisy fields, two shapes, `default`/`sparse` and three deltas. The slot size checks in `Prefetch` (#24) rely on this. |
| `slot_fits_rounding` | A Revolve + prefetching cycle through `Prefetch` with a `CompressionFactor` chosen so that the largest compressed fields are bigger than the nominal slot (field size / `CompressionFactor`) but fit its capacity (rounded up to 2 MiB). Every restored field must match the saved one within the error bound. |
| `slot_overflow_detected` | The same cycle with a capacity below the compressed size: the run must stop with `GPUZIP error [DEVICE=...] compression into the GPU cache` instead of writing past the slot and going on. |
| `meta_prefetch_bitcomp`, `meta_prefetch_raw`, `meta_chkptonly_bitcomp`, `meta_chkptonly_raw`, `meta_uniform_bitcomp` | Checkpoint metadata (`CheckpointMeta_t`): full Revolve or Uniform cycles, with the prefetching cache of 2 or `CheckpointOnly`, with Bitcomp or without a compressor. Every Save carries metadata that differs per timestep; on every restore the fields and the metadata must be those of that timestep, whatever path they took (cache hit, prefetch, host pool). One restore in three asks for no metadata, which is allowed, and its buffer must stay untouched. |
| `meta_miss_raw`, `meta_miss_bitcomp` | The cache-miss path of the prefetching cache, without a compressor or with Bitcomp. The cycles above never miss (the planner prefetches every restore), so these cases save timesteps 0, 1, 2 into a cache of 2 without `Dispatch` and retrieve them in reverse order: 2 hits, then a miss for timestep 0, whose fields and metadata come from the host pool. |
| `meta_absent_rejected`, `meta_len_mismatch`, `meta_buffer_mismatch` | Misuse of the checkpoint metadata must stop the run with a `GPUZIP error ... checkpoint metadata` message: a Retrieve with metadata of a checkpoint saved without it, two Saves with metadata of different lengths, a Retrieve into a buffer of another length. |
| `order_prefetch_raw`, `order_chkptonly_raw` | `Retrieve(timestep, curr, prev, stream)` without a compressor must not overwrite `curr` and `prev` before the work already queued on `stream` (#3). Before each restore a kernel on `stream` waits 100 ms and then copies `curr`; it must see the field from before the restore, and after the Retrieve the fields must be the checkpoint. The prefetching case goes through two cache hits and one miss; `CheckpointOnly` copies on `stream` itself. |

## Build and run

```bash
cmake -S tests -B build_tests -DCMAKE_CUDA_ARCHITECTURES=<arch, e.g. 70 or 90>
cmake --build build_tests -j 8
cd build_tests && ctest --output-on-failure
```

The build downloads Thrust 2.0.1 and nvCOMP 2.6.1 (the package used by
`standalone-example`). To test another nvCOMP, pass its extracted directory:
`-DNVCOMP_ROOT=/path/to/nvcomp-linux-x86_64-5.3.0.16_cuda12-archive` (nvCOMP 5.x
keeps the native Bitcomp headers under `include/nvcomp/native/`).

On a cluster, build on the login node and run `ctest` inside a GPU job.
