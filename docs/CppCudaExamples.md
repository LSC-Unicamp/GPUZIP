# Using with C++ and CUDA

This document provides detailed instructions on how to integrate GPUZIP with C++ and CUDA projects. It covers the necessary dependencies, how to include GPUZIP in your CMakeLists.txt, and how to manage GPUZIP's CMake flags. Additionally, an example of multi-GPU adjoint computing using GPUZIP is provided to demonstrate its practical application. Follow the steps outlined in this guide to leverage GPUZIP's capabilities in your own projects.

## Using GPUZIP in C++/CUDA projects (CMakeLists.txt)

### Dependencies (versions below were tested).
- cmake 3.22
- make 4.2
- nvcc 10.1
- CUDA 12.2
- cuZFP -- See `Installing cuZFP` tutorial.

Or use our Docker image `docker pull maltempi/awave-dev:ompc`

### Including GPUZIP stuff in your CMakeLists.txt 

```cmake
  set(CMAKE_CXX_STANDARD 17)

  include_directories(./GPUZIP/src/Prefetch/include)
  include_directories(./GPUZIP/src/Compressor/include)
  add_library(cuda_utils ./GPUZIP/src/Compressor/cuda_utils.cu)
  
  option(USE_NVTX "GPUZIP use NVTX (nsight) tracing." OFF)
  if (USE_NVTX)
      add_definitions(-DUSE_NVTX)
  endif()

  # Enable ZFP compression
  option(ZFP "Includes cuZFP as an available compressor." ON)
  if (ZFP)
      add_definitions(-DZFP)
      include_directories(/opt/zfp/include)

      # Ensure CMAKE_CUDA_FLAGS is defined before appending
      if (NOT DEFINED CMAKE_CUDA_FLAGS)
          set(CMAKE_CUDA_FLAGS "")
      endif()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")
  
      target_link_libraries(awave3d-decom
          /opt/zfp/lib/libzfp.so
          /opt/zfp/lib/libzfp.so.1
          /opt/zfp/lib/libzfp.so.1.0.0
          -lnvToolsExt
          -lcuda
          -lcusparse)
  endif()
  
  # Enable NVCOMP BITCOMP
  option(NVCOMP_BITCOMP "Includes NVIDIA Bitcomp as an available compressor." ON)
  if (NVCOMP_BITCOMP)
      add_definitions(-DNVCOMP_BITCOMP)
      message(STATUS "Using NVCOMP BITCOMP")
  
      include(FetchContent)
  
      FetchContent_Declare(
          NVCOMP_BITCOMP
          DOWNLOAD_EXTRACT_TIMESTAMP false
          URL https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
          URL_HASH SHA256=ac4834397291f245578af959694e816d96f80036eac50b5f24b113dee5b54225
          TLS_VERIFY false
      )
      FetchContent_MakeAvailable(NVCOMP_BITCOMP)
  
      FetchContent_GetProperties(NVCOMP_BITCOMP SOURCE_DIR NVCOMP_SRC_DIR)
      message(STATUS "NVCOMP_SRC_DIR: ${NVCOMP_SRC_DIR}")
  
      # Properly update CMAKE_PREFIX_PATH
      list(APPEND CMAKE_PREFIX_PATH "${NVCOMP_SRC_DIR}/lib/cmake/nvcomp")
  
      message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
  
      find_package(nvcomp REQUIRED CONFIG PATHS "${NVCOMP_SRC_DIR}/lib/cmake/nvcomp")
  
      target_link_libraries(awave3d-decom
          nvcomp::nvcomp_bitcomp
          cuda_utils
          -lnvToolsExt
          -lcuda
          -lcusparse)
  endif()
  
  # Enable cuSZp compression
  option(CUSZP "Includes cuSZp as an available compressor." ON)

  IF (CUSZP)
    add_definitions(-DCUSZP)
    message("Using CUSZP")
    include(FetchContent)

    FetchContent_Declare(
        cuszp
        GIT_REPOSITORY https://github.com/szcompressor/cuSZp.git
        GIT_TAG cuSZp-V1.1
    )
    FetchContent_MakeAvailable(cuszp)
    target_link_libraries(awave3d-decom
      cuSZp
      cuda_utils
      -lnvToolsExt
      -lcuda
      -lcusparse)
  ENDIF()
```

### Managing GPUZIP's cmake flags

By default, GPUZIP includes all compressors. However, once you choose the best compressor for your needs, you can build only the chosen one for production. You can disable specific compressors by setting the `CUZFP`, `CUSZP`, and `BITCOMP` flags to `0`. Another important flag is `USE_NVTX`, which enables internal GPUZIP NVTX flags to be visualized and measured on NSight.

```sh
## Default -- FULL (all compressors are enabled by default on this configuration)
cmake 

## Disabling cuZFP and cuSZP
cmake -DCUZFP=0 -DCUSZP=0

## Disabling Bitcomp only
cmake -DBITCOMP=0

## Enabling NVTX
cmake -DUSE_NVTX=1
```

## EXAMPLE: Multi-GPU adjoint computing (Prefetch+Compression)

The following example can run regardless of the chosen compressor or checkpointing algorithm.

In this snippet, `your_data_t` is used as an example to represent the current and previous fields from the adjoint computation. Since this is a multi-GPU implementation, the fields are divided into pieces. Adapt the snippet to fit your own data structures.

All possible GPUZIP configurations can be found within the `gpuzip_config_t` struct, which is defined in `Prefetch/include/common/GPUZIPBuilders.cpp`.

```cpp
#include "prefetch/Prefetch.cuh"
#include "prefetch/Checkpointing.hpp"
#include "common/GPUZIPBuilders.cpp"
#include "common/GPUZIPConfig.h"

void adjoint(gpuzip_config_t *gpuzip_config, your_data_t *data, int num_gpus) {
    GPUZIPLogger::SetLevel(gpuzip_config->log_level);
    GPUZIPLogger::PerfTraceSwitch(gpuzip_config->enable_performance_log);

    int steps = afd->nt;

    bool useCompression = gpuzip_config->compressor > 0;

    Checkpointing *chkpt = CheckpointingBuilder(gpuzip_config, steps);
    int snaps = chkpt->GetNumberOfCheckpoints();

    Prefetch *prefetch[num_gpus];
    for (int d = 0; d < num_gpus; d++) {
        size_t n1 = std::max(data->curr->devices[d].n1, data->prev->devices[d].n1);
        size_t n2 = std::max(data->curr->devices[d].n2, data->prev->devices[d].n2);
        size_t n3 = std::max(data->curr->devices[d].n3, data->prev->devices[d].n3);
        prefetch[d] = PrefetchBuilder(gpuzip_config, steps, chkpt); 
    }

    for (int shot = 0; shot <= data->shots; shot++) {
        for (int d = 0; d < num_gpus; d++) {
            cudaSetDevice(d);
            prefetch[d]->Setup();
        }

        chkpt->Reset();
        bool terminate = false;

        do {

            Action action = chkpt->GetAction();

            for (int d = 0; d < num_gpus; d++) {
                cudaSetDevice(d);
                prefetch[d]->Dispatch(chkpt->GetIt());
            }

            if (action.actionType == ACTION_SAVE) {
                for (int d = 0; d < num_gpus; d++) {
                    cudaSetDevice(d);

                    Field_t curr;
                    curr.n1 = data->curr->devices[d].n1;
                    curr.n2 = data->curr->devices[d].n2;
                    curr.n3 = data->curr->devices[d].n3;
                    curr.data = data->curr->devices[d].data;
                    curr.size = curr.n1 * curr.n2 * curr.n3 * sizeof(float);

                    Field_t prev;
                    prev.n1 = data->prev->devices[d].n1;
                    prev.n2 = data->prev->devices[d].n2;
                    prev.n3 = data->prev->devices[d].n3;
                    prev.data = data->prev->devices[d].data;
                    prev.size = prev.n1 * prev.n2 * prev.n3 * sizeof(float);

                    // NOTE: here is the last chance to check if all computation has finished before save.
                    // Suggestion: use cudaStreamSynchronize() or cudaDeviceSynchronize()

                    if (useCompression) {
                        auto comp = CompressorBuilder(gpuzip_config, curr.n1, curr.n2, curr.n3);
                        auto compprev = CompressorBuilder(gpuzip_config, prev.n1, prev.n2, prev.n3);
                        prefetch[d]->Save(action.ts, &curr, &prev, comp.get(), compprev.get());
                    } else {
                        prefetch[d]->Save(action.ts, &curr, &prev);
                    }
                }

                if (action.actionType == ACTION_FORWARD) {
                    // Call CUDA kernel for Forward computation on the given timestep (action.ts)
                    forward_computation(action.ts, data);
                }

                if (action.actionType == ACTION_BACKWARD) {
                    // Call CUDA kernel for Backward computation on the given timestep (action.ts)
                    backward_computation(action.ts, data);
                }

                if (action.actionType == ACTION_RESTORE) {
                    for (int d = 0; d < num_gpus; d++) {
                        Field_t curr;
                        curr.data = data->curr->devices[d].data;
                        curr.n1 = data->curr->devices[d].n1;
                        curr.n2 = data->curr->devices[d].n2;
                        curr.n3 = data->curr->devices[d].n3;

                        Field_t prev;
                        prev.data = data->prev->devices[d].data;
                        prev.n1 = data->prev->devices[d].n1;
                        prev.n2 = data->prev->devices[d].n2;
                        prev.n3 = data->prev->devices[d].n3;

                        if (useCompression) {
                            auto comp = CompressorBuilder(gpuzip_config, curr.n1, curr.n2, curr.n3);
                            auto compprev = CompressorBuilder(gpuzip_config, prev.n1, prev.n2, prev.n3);
                            prefetch[d]->Retrieve(action.ts, &curr, &prev, comp.get(), compprev.get());
                        } else {
                            prefetch[d]->Retrieve(action.ts, &curr, &prev, streams[d].compute);
                        }
                    }
                }
            }

            // In case of error, GPUZIP will log the error message with fprintf(stderr)
            if (action.actionType == ACTION_TERMINATE || action.actionType == ACTION_ERROR) {
                terminate = true;
            }
        } while(!terminate);
    }

    for (int d = 0; d < num_gpus; d++) {
        prefetch[d]->Report(); // Optional: prints # of prefetched checkpoints and avoided misses.
        prefetch[d]->Free();
    }
}
```
