#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuSZp_utility.h>
#include <cuSZp_entry_f32.h>
#include <cuSZp_timer.h>
#include "compressor.hpp"
#include "cuda_utils.hpp"

template <typename decompressType, typename compressedType>
class compressor_cuszp final : public compressor<decompressType, compressedType>
{
public:
  explicit compressor_cuszp(const std::size_t n1, const std::size_t n2,
                            const std::size_t n3,
                            const double error_bound)
      : n1_(n1), n2_(n2), n3_(n3), n_(n1 * n2 * n3), error_bound_(error_bound) {}

protected:
  size_t compress(decompressType *buf_in, compressedType *buf_out) override
  {
    size_t compressed_size;
    double rel_errbound = error_bound_ * (maxFloat(reinterpret_cast<float *>(buf_in), n_) - minFloat(reinterpret_cast<float *>(buf_in), n_));
    SZp_compress_deviceptr_f32(
      reinterpret_cast<float *>(buf_in), 
      reinterpret_cast<unsigned char *>(buf_out), 
      n_, 
      &compressed_size,
      rel_errbound, 
      cudaStreamDefault
    );
    return compressed_size;
  }
  void decompress(compressedType *buf_in, decompressType *buf_out, size_t compressed_size = -1) override
  {
    SZp_decompress_deviceptr_f32(
      reinterpret_cast<float *>(buf_out),
      reinterpret_cast<unsigned char *>(buf_in),
      n_,
      compressed_size,
      error_bound_,
      cudaStreamDefault
    );
  }
  std::string name() override { return "cuszp compressor"; }
  std::string description() override { return "cuszp compressor"; }
  std::size_t compressed_buffer_size(compressedType *buf = nullptr) override
  {
    throw std::runtime_error("compressed_buffer_size is not supported on cuszp");
  }
  std::size_t compressed_buffer_max_size() override
  {
    // https://github.com/szcompressor/cuSZp/blob/f47064f4edbc00aceb36692232ac7eef3fefaf2b/examples/cuSZp_gpu_f32_api.cpp#L64
    return ((n_ + 262144 - 1) / 262144 * 262144) * sizeof(float);
  }

private:
  const double error_bound_;
  const size_t n1_;
  const size_t n2_;
  const size_t n3_;
  const size_t n_;
};
