cmake_minimum_required(VERSION 3.22)
project(gpuzip_tests LANGUAGES CXX CUDA)

# GPU tests for GPUZIP. They need an NVIDIA GPU at run time (see README.md).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)
include(FetchContent)
enable_testing()

set(GPUZIP_ROOT           "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(GPUZIP_PREFETCH_INC   "${GPUZIP_ROOT}/src/Prefetch/include")
set(GPUZIP_COMPRESSOR_INC "${GPUZIP_ROOT}/src/Compressor/include")

# Thrust (required by cuda_utils.cu for the reductions used by Bitcomp)
FetchContent_Declare(Thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG        2.0.1
)
FetchContent_MakeAvailable(Thrust)
thrust_create_target(Thrust)

add_library(gpuzip_cuda_utils STATIC "${GPUZIP_ROOT}/src/Compressor/cuda_utils.cu")
target_include_directories(gpuzip_cuda_utils PUBLIC
    "${GPUZIP_COMPRESSOR_INC}" "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(gpuzip_cuda_utils PUBLIC Thrust)

# nvCOMP (Bitcomp). By default the same package as standalone-example; pass
# -DNVCOMP_ROOT=<extracted nvCOMP> to test another one (e.g. nvCOMP 5.x).
set(NVCOMP_ROOT "" CACHE PATH "Extracted nvCOMP to use instead of downloading 2.6.1")
add_executable(test_slot_capacity test_slot_capacity.cu)
if(NVCOMP_ROOT)
    find_library(NVCOMP_LIB nvcomp PATHS "${NVCOMP_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    # nvCOMP 5.x keeps the native Bitcomp API under include/nvcomp/native/
    target_include_directories(test_slot_capacity PRIVATE
        "${NVCOMP_ROOT}/include" "${NVCOMP_ROOT}/include/nvcomp")
    target_link_libraries(test_slot_capacity PRIVATE "${NVCOMP_LIB}")
    set_target_properties(test_slot_capacity PROPERTIES BUILD_RPATH "${NVCOMP_ROOT}/lib")
    message(STATUS "nvCOMP: ${NVCOMP_LIB}")
else()
    FetchContent_Declare(nvcomp_fetch
        DOWNLOAD_EXTRACT_TIMESTAMP false
        URL      https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
        URL_HASH SHA256=ac4834397291f245578af959694e816d96f80036eac50b5f24b113dee5b54225
    )
    FetchContent_MakeAvailable(nvcomp_fetch)
    list(APPEND CMAKE_PREFIX_PATH "${nvcomp_fetch_SOURCE_DIR}/lib/cmake/nvcomp")
    find_package(nvcomp REQUIRED CONFIG)
    target_include_directories(test_slot_capacity PRIVATE "${nvcomp_fetch_SOURCE_DIR}/include")
    target_link_libraries(test_slot_capacity PRIVATE nvcomp::nvcomp)
    message(STATUS "nvCOMP: 2.6.1 (downloaded)")
endif()

target_include_directories(test_slot_capacity PRIVATE
    "${GPUZIP_PREFETCH_INC}" "${GPUZIP_COMPRESSOR_INC}" "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(test_slot_capacity PRIVATE gpuzip_cuda_utils CUDA::cudart)
target_compile_definitions(test_slot_capacity PRIVATE NVCOMP_BITCOMP)

add_test(NAME bitcomp_write_extent COMMAND test_slot_capacity bitcomp_write_extent)
add_test(NAME slot_fits_rounding COMMAND test_slot_capacity slot_fits_rounding)
add_test(NAME slot_overflow_detected COMMAND test_slot_capacity slot_overflow_detected)
# The run must stop with GPUZIP's message (exit code is not checked)
set_tests_properties(slot_overflow_detected PROPERTIES
    PASS_REGULAR_EXPRESSION "GPUZIP error \\[DEVICE=[0-9]+\\] compression into the GPU cache"
    FAIL_REGULAR_EXPRESSION "the run completed although")
set_tests_properties(bitcomp_write_extent slot_fits_rounding slot_overflow_detected
    PROPERTIES LABELS "gpu")
