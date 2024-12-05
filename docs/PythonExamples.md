# Using GPUZIP with Python

GPUZIP's `Compressor` component has been bound to Python.

The input data needs to be in `cupy` format, which is a NumPy-compatible format designed for CUDA. For more information, visit [CuPy's official website](https://cupy.dev/).

A complete example can be found in the GPUZIP_Compressors repository at `./GPUZIPy/example/main.py`.

> **Note:** Currently, there are no Python bindings available for GPUZIP's `Prefetch` component.

## Building

### Dependencies (versions tested):
- nvcc 10.1
- CUDA 12.2
- NVIDIA driver 536.19
- cmake 3.22 (required)
- make 4.2.0
- python 3.9

### Building the Python Package
```sh
git submodule update --init --recursive
cd GPUZIPy
pip install .
```

### Running the Example
```sh
cd GPUZIPy/example
pip install -r requirements.txt
python main.py
```

## Building with Docker

### Building the Docker Image
```sh
git submodule update --init --recursive
docker build . -f ./GPUZIPy/dockerfile.gpuzipy -t maltempi/gpuzipy:latest
```

To specify a particular CUDA version, use the `build-arg` option:
```sh
git submodule update --init --recursive
docker build . -t maltempi/gpuzipy:latest --build-arg CUDA_VERSION=12.1.0
```

### Running the Example with Docker
```sh
git submodule update --init --recursive
docker run --gpus all --rm --env NVIDIA_DISABLE_REQUIRE=1 maltempi/gpuzipy:latest
```