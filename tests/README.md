# GPUZIP tests

GPU tests (they need an NVIDIA GPU to run). A first step towards #9.

| Test | What it checks |
|------|----------------|
| `bitcomp_write_extent` | Bitcomp writes nothing past the compressed size it reports: the buffer is filled with a guard byte before compressing and every byte after that size must be unchanged. Also checks the decompression error bound. Covers zeros, smooth and noisy fields, two shapes, `default`/`sparse` and three deltas. The slot size checks in `Prefetch` (#24) rely on this. |
| `slot_fits_rounding` | A Revolve + prefetching cycle through `Prefetch` with a `CompressionFactor` chosen so that the largest compressed fields are bigger than the nominal slot (field size / `CompressionFactor`) but fit its capacity (rounded up to 2 MiB). Every restored field must match the saved one within the error bound. |
| `slot_overflow_detected` | The same cycle with a capacity below the compressed size: the run must stop with `GPUZIP error [DEVICE=...] compression into the GPU cache` instead of writing past the slot and going on. |

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
