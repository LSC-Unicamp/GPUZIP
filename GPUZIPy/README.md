# GPUZIPy

Python bindings for [GPUZIP](https://github.com/LSC-Unicamp/GPUZIP)'s GPU compression layer
(`Compressor`), exposing `CompressorBitcomp` (NVIDIA Bitcomp), `CompressorZFP` (cuZFP), and
`CompressorCuszp` (cuSZp) to Python via pybind11.

Input/output buffers are raw GPU device pointers (e.g. from [CuPy](https://cupy.dev/)).

> **Note:** Only the `Compressor` component is bound to Python. GPUZIP's `Prefetch`/checkpointing
> layer is C++/CUDA only.

## Requirements

This package has no prebuilt wheels: installing it compiles NVIDIA Bitcomp (nvcomp), cuZFP,
and cuSZp against your local CUDA toolkit.

- NVIDIA CUDA Toolkit (tested with CUDA 12.2, nvcc 10.1)
- CMake >= 3.22
- Internet access at install time (build-time dependencies are fetched automatically: Thrust, ZFP, cuSZp, nvcomp)

## Run the examples with Docker

```sh
# from the repo root
mkdir -p GPUZIPy/example/output
docker run --rm --gpus all \
    -v $PWD/GPUZIPy/example/output:/GPUZIP/GPUZIPy/example/output \
    maltempi/gpuzipy:latest
```

The container is removed after it exits (`--rm`), so without that volume mount the
rendered PSNR/SSIM slice comparisons in `example/output/` would be lost along with it.
With it, results land directly in `GPUZIPy/example/output/` on the host.

## Install

```sh
pip install gpuzipy
```

## Usage

```python
import gpuzipy

comp = gpuzipy.CompressorBitcomp(n1, n2, n3, config_kind, range_fraction, num_sigma, delta, "float", "default")
max_buf = gpuzipy.compressed_buffer_max_size(comp)
size = gpuzipy.compress(comp, d_uncompressed_ptr, d_compressed_ptr)
gpuzipy.decompress(comp, d_compressed_ptr, d_uncompressed_ptr, size)
```

See [`example/main.py`](https://github.com/LSC-Unicamp/GPUZIP/blob/main/GPUZIPy/example/main.py)
for a full round-trip example, and the
[Python usage docs](https://github.com/LSC-Unicamp/GPUZIP/blob/main/docs/PythonExamples.md).

## License

MIT
