# GPUZIPy

Test GPUZIPy in Colab:
[![Test GPUZIPy In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSC-Unicamp/GPUZIP/blob/main/GPUZIPy/example/GPUZIPy_Example.ipynb) 

Python bindings for [GPUZIP](https://github.com/LSC-Unicamp/GPUZIP)'s GPU compression layer
(`Compressor`), exposing `CompressorBitcomp` (NVIDIA Bitcomp), `CompressorZFP` (cuZFP), and
`CompressorCuszp` (cuSZp) to Python via pybind11.

Input/output buffers are raw GPU device pointers (e.g. from [CuPy](https://cupy.dev/)).

> **Note:** Only the `Compressor` component is bound to Python. GPUZIP's `Prefetch`/checkpointing layer is C++/CUDA only.

## Install

Prebuilt wheels (Linux x86_64, Python 3.10-3.13, CUDA 12, GPUs from Volta/sm_70 to Hopper/sm_90)
are attached to the [GitHub Releases](https://github.com/LSC-Unicamp/GPUZIP/releases):

```sh
pip install gpuzipy --only-binary gpuzipy \
    --find-links https://github.com/LSC-Unicamp/GPUZIP/releases/expanded_assets/gpuzipy-v2.1.0
```

pip picks the wheel matching your Python version. `--only-binary` stops it from picking
GitHub's auto-generated source archive listed on the same page.

Only an NVIDIA driver supporting CUDA 12 is needed; nvcomp (Bitcomp), cuZFP and cuSZp are bundled.

### Build from source

Compiles NVIDIA Bitcomp (nvcomp), cuZFP, and cuSZp against your local CUDA toolkit.

- NVIDIA CUDA and NVCC (tested with CUDA 12.2)
- CMake >= 3.22
- Internet access at install time (build-time dependencies are fetched automatically: Thrust, ZFP, cuSZp, nvcomp)

```sh
pip install .
```

Set `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native"` to compile only for the local GPU.

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

## Releasing wheels

`.github/workflows/gpuzipy-wheels.yml` builds the wheels with cibuildwheel (config in
`pyproject.toml`). Bump `version` in `setup.py`, then push a matching tag, e.g.
`git tag gpuzipy-v2.1.1 && git push origin gpuzipy-v2.1.1`, to publish a GitHub Release with
the wheels. Then update the `--find-links` tag above and in the example notebook.

## License

MIT
