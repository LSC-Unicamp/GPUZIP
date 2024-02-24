# GPUZIP

GPUZIP is a library designed to enhance checkpointing on GPUs by combining data compression and prefetching techniques. It leverages the Revolve checkpoint algorithm and GPU-based data compression to optimize memory utilization and improve computation-to-communication ratios.

Supported compressors: [NVIDIA Bitcomp](https://developer.nvidia.com/nvcomp), [cuSZp](https://dl.acm.org/doi/10.1145/3581784.3607048), and [cuZFP](https://zfp.readthedocs.io/en/release1.0.1/execution.html#using-cuda).

[Learn more](https://doi.org/10.1007/978-3-031-69583-4_12)

## GPUZIPy

### Building
#### Dependencies (versions below were tested):
- nvcc 10.1
- CUDA 12.2
- NVidia driver 536.19
- cmake 3.22 (required)
- make 4.2.0
- python 3.9

```sh
cd GPUZIPy
pip install .
```

### Building with Docker
```sh
docker build . -f ./GPUZIPy/dockerfile.gpuzipy -t maltempi/gpuzipy:latest
```

If you want to have a docker image with an specific Cuda version, just specify it in the `build-arg`.

```sh
docker build . -t maltempi/gpuzipy:latest  --build-arg CUDA_VERSION=12.1.0
```

### Running example
```sh
docker run --gpus all --rm --env NVIDIA_DISABLE_REQUIRE=1 maltempi/gpuzipy:latest
```

## Cite

> Thiago Maltempi, Sandro Rigo, Marcio Pereira, Hervé Yviquel, Jessé Costa, and Guido Araujo. 2024. Combining Compression and Prefetching to Improve Checkpointing for Inverse Seismic Problems in GPUs. In Euro-Par 2024: Parallel Processing: 30th European Conference on Parallel and Distributed Processing, Madrid, Spain, August 26–30, 2024, Proceedings, Part III. Springer-Verlag, Berlin, Heidelberg, 167–181. https://doi.org/10.1007/978-3-031-69583-4_12

```text
@inproceedings{10.1007/978-3-031-69583-4_12,
    author = {Maltempi, Thiago and Rigo, Sandro and Pereira, Marcio and Yviquel, Herv\'{e} and Costa, Jess\'{e} and Araujo, Guido},
    title = {Combining Compression and&nbsp;Prefetching to&nbsp;Improve Checkpointing for&nbsp;Inverse Seismic Problems in&nbsp;GPUs},
    year = {2024},
    isbn = {978-3-031-69582-7},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    url = {https://doi.org/10.1007/978-3-031-69583-4_12},
    doi = {10.1007/978-3-031-69583-4_12},
    abstract = {Inverse problems are crucial in various scientific and engineering fields requiring intricate mathematical and computational modeling. An example of such a problem is the Full Waveform Inversion (FWI), used in several geophysical applications like oil reservoir discovery. Central to solving FWI is Reverse Time Migration (RTM), a Geophysical algorithm for high-resolution subsurface imaging from seismic data that poses considerable computational challenges due to its extensive memory and computation demands. A typical approach to address the memory constraints of RTM includes decomposing the processing tasks in multiple GPUs, checkpointing the intermediate results, and rematerializing the computation from checkpoints when needed. This paper introduces a novel checkpoint prefetching mechanism called GPUZIP. It combines Revolve, a well-known checkpoint algorithm, and GPU-based data compression to improve checkpoint memory utilization. GPUZIP was designed to allow the flexible utilization of different compression algorithms and target applications. Experimental results show that the combination of prefetching and GPU data compression enabled by GPUZIP significantly improves the computation-to-communication ratio for the RTM application. Speed-ups of up to 3.90\texttimes{} and a remarkable 80\texttimes{} Host-to-Device data transfer reduction have been achieved when running a well-known geophysics benchmark. The proposed approach mitigates the computational challenges of RTM and has the potential for applicability and to bring performance improvements in other scientific computing fields.},
    booktitle = {Euro-Par 2024: Parallel Processing: 30th European Conference on Parallel and Distributed Processing, Madrid, Spain, August 26–30, 2024, Proceedings, Part III},
    pages = {167–181},
    numpages = {15},
    keywords = {High-Performance Computing, Data compression, Reverse Time Migration, Prefetching, Checkpointing},
    location = {Madrid, Spain}
}
```
