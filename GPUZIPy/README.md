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

## Build

With cudnn
```sh
docker pull nvidia/cuda:13.9.2-cudnn-devel-ubuntu24.04
docker run -it -v $PWD/..:/GPUZIP --gpus all --rm nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04 bash
cd /GPUZIP/GPUZIPy
apt-get update
apt-get install python3 python3.12-venv python3-dev git -y
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

with cupy:

```sh
docker run -it -v $PWD/..:/GPUZIP --gpus all --rm cupy/cupy:v13.6.0 bash
cd /GPUZIP/GPUZIPy
apt-get update
apt install git -y
pip install .
python example/main.py
```

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
