#pragma once
#include "Compressor.hpp"
#include "cuda_utils.hpp"
#include "cuSZp.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * @class CompressorCuSZp
 * @brief A compressor implementation using cuSZp for floating-point data
 * compression on GPUs.
 *
 * This class provides methods for compressing and decompressing floating-point
 * data using the cuSZp library. It supports device pointers and utilizes CUDA
 * streams for execution.
 *
 * @tparam decompressType The data type used for decompression (typically
 * float).
 * @tparam compressedType The data type used for compressed storage (typically
 * unsigned char).
 *
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date November 25, 2023
 */
template <typename decompressType, typename compressedType>
class CompressorCuSZp final
    : public Compressor<decompressType, compressedType> {
public:
  /**
   * @brief Constructs a cuSZp compressor with the given dimensions and error
   * bound.
   *
   * @param n1 First dimension of the input data.
   * @param n2 Second dimension of the input data.
   * @param n3 Third dimension of the input data.
   * @param float_kind The kind of floating-point data ('float' or 'double').
   * @param error_bound The error bound for lossy compression.
   */
  explicit CompressorCuSZp(const std::size_t n1, 
                           const std::size_t n2,
                           const std::size_t n3, 
                           const double error_bound,
                           const std::string &float_kind = "float")
      : n1_(n1), n2_(n2), n3_(n3), n_(n1 * n2 * n3), error_bound_(error_bound), float_kind_(float_kind) 
  {
    if (float_kind_ != "float" && float_kind_ != "double")
    {
      throw std::invalid_argument("float_kind must be 'float' or 'double'");
    }
  }

protected:
  /**
   * @brief Compresses a floating-point buffer using cuSZp.
   *
   * @param buf_in Pointer to the input data buffer (device memory).
   * @param buf_out Pointer to the output compressed data buffer (device
   * memory).
   * @return The size of the compressed data in bytes.
   */
  size_t compress(decompressType *buf_in, compressedType *buf_out) override {
    size_t compressed_size;
    if (float_kind_ == "float")
    {
      float rel_errbound = error_bound_ *
                          (maxFloat(reinterpret_cast<float *>(buf_in), n_) -
                          minFloat(reinterpret_cast<float *>(buf_in), n_));

      cuSZp_compress_1D_fixed_f32(reinterpret_cast<float *>(buf_in),
                                  reinterpret_cast<unsigned char *>(buf_out), n_,
                                  &compressed_size, rel_errbound,
                                  cudaStreamDefault);
    }
    else if (float_kind_ == "double")
    {
      double rel_errbound = error_bound_ *
                          (maxDouble(reinterpret_cast<double *>(buf_in), n_) -
                          minDouble(reinterpret_cast<double *>(buf_in), n_));

      cuSZp_compress_1D_fixed_f64(reinterpret_cast<double *>(buf_in),
                                  reinterpret_cast<unsigned char *>(buf_out), n_,
                                  &compressed_size, rel_errbound,
                                  cudaStreamDefault);
    }
    return compressed_size;
  }

  /**
   * @brief Decompresses a cuSZp-compressed buffer.
   *
   * @param buf_in Pointer to the compressed data buffer (device memory).
   * @param buf_out Pointer to the output decompressed data buffer (device
   * memory).
   * @param compressed_size The size of the compressed input buffer (default:
   * -1, not used in cuSZp).
   */
  void decompress(compressedType *buf_in, decompressType *buf_out,
                  size_t compressed_size = -1) override {
    if (float_kind_ == "float")
    {
        cuSZp_decompress_1D_fixed_f32(reinterpret_cast<float *>(buf_out),
                                     reinterpret_cast<unsigned char *>(buf_in), n_,
                                     compressed_size, error_bound_,
                                     cudaStreamDefault);
    }
    else if (float_kind_ == "double")
    {
        cuSZp_decompress_1D_fixed_f64(reinterpret_cast<double *>(buf_out),
                                      reinterpret_cast<unsigned char *>(buf_in), n_,
                                      compressed_size, error_bound_,
                                      cudaStreamDefault);
    }
    else
    {
      throw std::invalid_argument("invalid argument for float_kind");
    }
  }

  /**
   * @brief Gets the name of the compressor.
   *
   * @return The name of the compressor as a string.
   */
  std::string name() override { return "cuszp compressor"; }

  /**
   * @brief Gets a brief description of the compressor.
   *
   * @return A string describing the compressor.
   */
  std::string description() override { return "cuszp compressor"; }

  /**
   * @brief Gets the size of the compressed buffer. **Not supported for cuSZp.**
   *
   * @throws std::runtime_error Always throws an exception since this function
   * is not implemented.
   */
  std::size_t compressedBufferSize(compressedType *buf = nullptr) override {
    throw std::runtime_error(
        "compressedBufferSize is not supported on cuszp");
  }

  /**
   * @brief Gets the maximum possible size of a compressed buffer.
   *
   * The maximum size is calculated based on cuSZp's padding strategy.
   *
   * @return The maximum compressed buffer size in bytes.
   */
  std::size_t compressedMaxSize() override {
    // https://github.com/szcompressor/cuSZp/blob/f47064f4edbc00aceb36692232ac7eef3fefaf2b/examples/cuSZp_gpu_f32_api.cpp#L64
    size_t elem_size;

    if (float_kind_ == "float")
        elem_size = sizeof(float);
    else if (float_kind_ == "double")
        elem_size = sizeof(double);
    else
        throw std::invalid_argument("invalid argument for float_kind");

    return ((n_ + 262144 - 1) / 262144 * 262144) * elem_size;
  }

private:
  std::string float_kind_;
  const double error_bound_; ///< Absolute error bound for compression.
  const size_t n1_;          ///< First dimension of input data.
  const size_t n2_;          ///< Second dimension of input data.
  const size_t n3_;          ///< Third dimension of input data.
  const size_t n_;           ///< Total number of elements in the input data.
};
