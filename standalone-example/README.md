# GPUZIP Standalone Example

End-to-end demo of the GPUZIP library on a **dummy 3-D acoustic wave adjoint simulation**.

- Forward pass propagates a Ricker-wavelet source forward in time.
  GPUZIP saves checkpoints as directed by the checkpointing algorithm.
- Backward pass propagates an adjoint wavefield and accumulates the
  cross-correlation imaging condition, restoring forward checkpoints
  from the GPUZIP prefetch cache or host memory as needed.

GPUZIP headers are included directly as source — **no pre-compilation
of the library is required**.

---

## Requirements

| Requirement | Version tested |
|-------------|---------------|
| CMake       | ≥ 3.22        |
| CUDA toolkit | 11.2 – 12.x  |
| NVIDIA GPU  | Required at runtime |
| C++ compiler | C++17 (e.g. GCC 9+, Clang 15) |
| Internet access | FetchContent downloads ZFP, nvcomp, cuSZp |

Docker image with all dependencies pre-installed: `maltempi/awave-dev:ompc`

---

## Running with Docker

The Docker image ships CMake, CUDA toolkit, compilers, and all library dependencies.
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

**Pull:**
```bash
docker pull maltempi/awave-dev:ompc
```

**Run interactively with GPU access, mounting the repo:**
```bash
docker run --rm -it --gpus all \
    -v $PWD:/workspace/GPUZIP \
    -w /workspace/GPUZIP/standalone-example \
    maltempi/awave-dev:ompc \
    bash
```

Replace `/path/to/GPUZIP` with the absolute path to this repository on your host.

**Inside the container — build and run:**
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./gpuzip_example
```

**One-liner (build + run without interactive shell):**
```bash
docker run --rm --gpus all \
    -v $PWD:/workspace/GPUZIP \
    -w /workspace/GPUZIP/standalone-example \
    maltempi/awave-dev:ompc \
    bash -c "mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc) && ./gpuzip_example"
```

---

## Build

```bash
cd standalone-example
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This builds `gpuzip_example` with all three compressor backends enabled
(cuZFP, Bitcomp, cuSZp). The default `config.cfg` is copied to the build
directory automatically.

### Disable individual backends

```bash
cmake .. -DWITH_ZFP=OFF -DWITH_CUSZP=OFF   # Bitcomp only
cmake .. -DWITH_BITCOMP=OFF                 # ZFP + cuSZp only
cmake .. -DWITH_ZFP=OFF -DWITH_BITCOMP=OFF -DWITH_CUSZP=OFF  # no compression
```

### Enable NVTX profiling (NSight)

```bash
cmake .. -DWITH_NVTX=ON
```

---

## Run

```bash
# From the build directory (config.cfg is copied here by cmake)
./gpuzip_example

# Or pass a custom config file
./gpuzip_example /path/to/my_config.cfg
```

Expected output:
```
=== GPUZIP Standalone Example ===
Grid        : 64 x 64 x 64
Steps       : 300
Checkpointing: Revolve
Cache capacity: 4
Compressor  : 2 (Bitcomp)
Snapshots   : 7

Running PSA (prefetch setup algorithm)... done.

Elapsed         : 3.21 s
Gradient L1 sum : 1.45e-07  (non-zero — simulation ran)

--- GPUZIP Cache Report ---
...
```

---

## Configuration

Edit `config.cfg` (or pass a custom path) to change simulation parameters.

```
# Grid dimensions
n1    = 64      # try 128x128x128 for a heavier run
n2    = 64
n3    = 64
steps = 300

# Checkpointing algorithm: 1=Revolve (recommended), 0=Trace
checkpointing_algorithm = 1

# Cache capacity: 0=no prefetch, >=2=LRU prefetch cache
cache_capacity = 4

# Compressor: 0=None, 1=cuZFP, 2=Bitcomp, 3=cuSZp
compressor = 2

# Bitcomp delta (error bound); lower = more accurate, less compression
bitcomp_delta = 1e-8

# Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
log_level = 1
```

### Key configuration options

| Option | Effect |
|--------|--------|
| `compressor = 0` | No compression; checkpoints transferred raw |
| `compressor = 1` + `zfp_bit_rate = 8` | cuZFP, ~4× compression ratio |
| `compressor = 2` + `bitcomp_delta = 1e-8` | Bitcomp, 100–300× ratio (data-dependent) |
| `cache_capacity = 0` | Disable prefetching (synchronous H↔D transfers) |
| `cache_capacity = 4` | LRU cache with 4 GPU slots (recommended for Revolve) |
| `compression_factor = 2.6` | Skip warm-up; pre-specify worst-case compression ratio |

---

## What the simulation does

```
Forward pass (GPUZIP-directed):
  ┌──────────────────────────────────────────┐
  │  for each ACTION_FORWARD:                │
  │      wave_step_kernel(fwd_next,          │
  │                        fwd_curr, fwd_prev│
  │                        + Ricker source)  │
  │      fwd_prev ← fwd_curr ← fwd_next      │
  │                                          │
  │  for each ACTION_SAVE:                   │
  │      GPUZIP saves (fwd_curr, fwd_prev)   │
  │      → GPU cache → host pinned memory    │
  └──────────────────────────────────────────┘

Backward pass (GPUZIP-directed):
  ┌──────────────────────────────────────────┐
  │  for each ACTION_RESTORE:                │
  │      GPUZIP retrieves checkpoint         │
  │      → GPU cache (or host → GPU)         │
  │      → fwd_curr, fwd_prev               │
  │                                          │
  │  for each ACTION_BACKWARD:               │
  │      gradient += fwd_curr * adj_curr     │  ← imaging condition
  │      wave_step_kernel(adj_next,          │
  │                        adj_curr, adj_prev│
  │                        + receiver inject)│
  │      adj_prev ← adj_curr ← adj_next      │
  └──────────────────────────────────────────┘
```

The gradient L1 sum printed at the end is a sanity check: a non-zero value confirms both passes ran and the imaging condition was applied.
