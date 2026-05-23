cmake_minimum_required(VERSION 3.22)
project(gpuzip_standalone_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

# GPUZIP is used as a header-only source library — no pre-compilation needed.
set(GPUZIP_ROOT           "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(GPUZIP_PREFETCH_INC   "${GPUZIP_ROOT}/src/Prefetch/include")
set(GPUZIP_COMPRESSOR_INC "${GPUZIP_ROOT}/src/Compressor/include")

# --------------------------------------------------------------------------
# Thrust  (required by cuda_utils.cu for GPU reductions used by Bitcomp)
# --------------------------------------------------------------------------
FetchContent_Declare(Thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG        2.0.1
)
FetchContent_MakeAvailable(Thrust)
thrust_create_target(Thrust)

# cuda_utils.cu: GPU min/max/mean/std reductions — compiled once as a library
add_library(gpuzip_cuda_utils STATIC
    "${GPUZIP_ROOT}/src/Compressor/cuda_utils.cu")
target_include_directories(gpuzip_cuda_utils PUBLIC
    "${GPUZIP_COMPRESSOR_INC}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(gpuzip_cuda_utils PUBLIC Thrust)

# --------------------------------------------------------------------------
# Compressor backends  (all enabled by default; disable with -DWITH_X=OFF)
# --------------------------------------------------------------------------
option(WITH_ZFP     "Enable cuZFP compressor"  ON)
option(WITH_BITCOMP "Enable Bitcomp compressor" ON)
option(WITH_CUSZP   "Enable cuSZp compressor"   OFF)
option(WITH_NVTX    "Enable NVTX profiling"     OFF)

# ---- cuZFP ----
if(WITH_ZFP)
    set(ZFP_WITH_CUDA ON CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")
    FetchContent_Declare(zfp_fetch
        GIT_REPOSITORY https://github.com/LLNL/zfp.git
        GIT_TAG        1.0.0
    )
    FetchContent_MakeAvailable(zfp_fetch)
endif()

# ---- Bitcomp (via nvcomp) ----
if(WITH_BITCOMP)
    FetchContent_Declare(nvcomp_fetch
        DOWNLOAD_EXTRACT_TIMESTAMP false
        URL      https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
        URL_HASH SHA256=ac4834397291f245578af959694e816d96f80036eac50b5f24b113dee5b54225
        TLS_VERIFY false
    )
    FetchContent_MakeAvailable(nvcomp_fetch)
    # nvcomp ships its own CMake config; point find_package at it
    list(APPEND CMAKE_PREFIX_PATH "${nvcomp_fetch_SOURCE_DIR}/lib/cmake/nvcomp")
    find_package(nvcomp REQUIRED CONFIG)
endif()

# ---- cuSZp ----
if(WITH_CUSZP)
    FetchContent_Declare(cuszp_fetch
        GIT_REPOSITORY https://github.com/szcompressor/cuSZp.git
        GIT_TAG        cuSZp-V1.1
    )
    FetchContent_MakeAvailable(cuszp_fetch)
endif()

# --------------------------------------------------------------------------
# Main executable
# --------------------------------------------------------------------------
add_executable(gpuzip_example main.cu)

target_include_directories(gpuzip_example PRIVATE
    "${GPUZIP_PREFETCH_INC}"
    "${GPUZIP_COMPRESSOR_INC}"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}")   # config_reader.hpp, kernels.cuh

target_link_libraries(gpuzip_example PRIVATE
    gpuzip_cuda_utils
    CUDA::cudart)

# Per-backend includes, libraries, and compile definitions
if(WITH_ZFP)
    target_include_directories(gpuzip_example PRIVATE "${zfp_fetch_SOURCE_DIR}/include")
    target_link_libraries(gpuzip_example PRIVATE zfp::zfp)
    target_compile_definitions(gpuzip_example PRIVATE ZFP)
endif()

if(WITH_BITCOMP)
    target_include_directories(gpuzip_example PRIVATE "${nvcomp_fetch_SOURCE_DIR}/include")
    target_link_libraries(gpuzip_example PRIVATE nvcomp::nvcomp)
    target_compile_definitions(gpuzip_example PRIVATE NVCOMP_BITCOMP)
endif()

if(WITH_CUSZP)
    target_link_libraries(gpuzip_example PRIVATE cuszp::cuszp)
    target_compile_definitions(gpuzip_example PRIVATE CUSZP)
endif()

if(WITH_NVTX)
    target_compile_definitions(gpuzip_example PRIVATE USE_NVTX)
    target_link_libraries(gpuzip_example PRIVATE -lnvToolsExt)
endif()

# Copy default config.cfg to build dir so the binary finds it at runtime
configure_file(config.cfg "${CMAKE_CURRENT_BINARY_DIR}/config.cfg" COPYONLY)

# --------------------------------------------------------------------------
# Build summary
# --------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== gpuzip_standalone_example ===")
message(STATUS "  WITH_ZFP     : ${WITH_ZFP}")
message(STATUS "  WITH_BITCOMP : ${WITH_BITCOMP}")
message(STATUS "  WITH_CUSZP   : ${WITH_CUSZP}")
message(STATUS "  WITH_NVTX    : ${WITH_NVTX}")
message(STATUS "")
