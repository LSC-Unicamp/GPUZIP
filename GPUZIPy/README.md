# GPUZIPy

Test GPUZIPy in Colab:
[![Test GPUZIPy In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSC-Unicamp/GPUZIP/blob/main/GPUZIPy/example/GPUZIPy_Example.ipynb) 

Python bindings for [GPUZIP](https://github.com/LSC-Unicamp/GPUZIP)'s GPU compression layer
(`Compressor`), exposing `CompressorBitcomp` (NVIDIA Bitcomp), `CompressorZFP` (cuZFP), and
`CompressorCuszp` (cuSZp) to Python via pybind11.

Input/output buffers are raw GPU device pointers (e.g. from [CuPy](https://cupy.dev/)).

> **Note:** Only the `Compressor` component is bound to Python. GPUZIP's `Prefetch`/checkpointing layer is C++/CUDA only.

## Requirements

This package has no prebuilt wheels: installing it compiles NVIDIA Bitcomp (nvcomp), cuZFP,
and cuSZp against your local CUDA toolkit.

- NVIDIA CUDA and NVCC (nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Tue_Aug_15_22:02:13_PDT_2023 Cuda compilation tools, release 12.2, V12.2.140 Build cuda_12.2.r12.2/compiler.33191640_0)
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
pip install .
```

## License

MIT
