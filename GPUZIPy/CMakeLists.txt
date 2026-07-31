cmake_minimum_required(VERSION 3.22)
project(gpuzip LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
message("CMAKE_PREFIX_PATH:${CMAKE_PREFIX_PATH}")

##
## Thrust dependency
##
FetchContent_Declare(
    THRUST
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG 2.0.1
)
FetchContent_MakeAvailable(THRUST)
thrust_create_target(Thrust)
find_package(Thrust REQUIRED CONFIG)

##
## cuZFP / zfp dependency
##
set(ZFP_WITH_CUDA ON CACHE BOOL "Build ZFP with CUDA support")
set(BUILD_CUDA ON CACHE BOOL "Build ZFP with CUDA support")
FetchContent_Declare(
    ZFP
    GIT_REPOSITORY https://github.com/LLNL/zfp.git
    GIT_TAG 1.0.0
)
FetchContent_MakeAvailable(ZFP)

##
## cuszp
##
set(ZFP_WITH_CUDA ON CACHE BOOL "Build cuszp with CUDA support")
set(BUILD_CUDA ON CACHE BOOL "Build cuszp with CUDA support")
FetchContent_Declare(
    cuszp
    GIT_REPOSITORY https://github.com/szcompressor/cuSZp.git
    GIT_TAG cuSZp-V3.0.0
)
FetchContent_MakeAvailable(cuszp)

##
## NVCOMP Bitcomp dependency
##
FetchContent_Declare(
    NVCOMP
    DOWNLOAD_EXTRACT_TIMESTAMP false
    URL https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
    URL_HASH SHA256=ac4834397291f245578af959694e816d96f80036eac50b5f24b113dee5b54225
    TLS_VERIFY false
)
FetchContent_MakeAvailable(NVCOMP)

FetchContent_GetProperties(NVCOMP SOURCE_DIR NVCOMP_SRC_DIR)
message("NVCOMP_SRC_DIR:${NVCOMP_SRC_DIR}")
if (NOT DEFINED ${CMAKE_PREFIX_PATH})
  set (CMAKE_PREFIX_PATH ${NVCOMP_SRC_DIR}/lib/cmake/nvcomp)
else()
  list(APPEND CMAKE_PREFIFX_PATH ${NVCOMP_SRC_DIR}/lib/cmake/nvcomp)
endif()

find_package(nvcomp REQUIRED)

##
## PyBind11 dependency
##
# Resolved via the pip-installed pybind11 package (setup.py passes
# -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")),
# so building an sdist does not require the pybind11 git submodule to be checked out.
find_package(pybind11 CONFIG REQUIRED)
include_directories(${ZFP_SOURCE_DIR}/include ${NVCOMP_SRC_DIR}/include)

# cuda_utils.cu: GPU min/max/mean/std reductions used by CompressorBitcomp/CompressorCuSZp.
# Declared in src/Compressor/include/cuda_utils.hpp but defined separately, so it must be
# compiled and linked in explicitly (it is not part of the header-only Compressor include tree).
add_library(gpuzip_cuda_utils STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Compressor/cuda_utils.cu)
target_include_directories(gpuzip_cuda_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Compressor/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gpuzip_cuda_utils PUBLIC Thrust)

pybind11_add_module(gpuzipy ./csrc/gpuzipy.cpp)
target_include_directories(gpuzipy PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# cuSZp_shared, not cuSZp_static: cuSZp is built with relocatable device code, and
# gpuzipy (a pybind11 MODULE target with no .cu sources of its own) can't reliably
# get CMake's device-link step to resolve it. cuSZp_shared is a real CUDA SHARED
# library, so its device code is already resolved when it's built, same as zfp's.
target_link_libraries(gpuzipy
  PRIVATE
  gpuzip_cuda_utils
  zfp::zfp
  cuSZp_shared
  nvcomp::nvcomp
  -lcuda
  -lcusparse)

target_compile_definitions(gpuzipy
                          PRIVATE VERSION_INFO=${VERSION_INFO})
