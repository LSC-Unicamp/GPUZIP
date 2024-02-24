#pragma once
#include <array>
#include <exception>
#include <stdexcept>
#include <memory>
#include <native/bitcomp.h>
#include <string>

#include "compressor.hpp"

#include "cuda_utils.hpp"

struct bitcompPlan
{
  explicit bitcompPlan(const std::size_t n, bitcompDataType_t dataType,
                       bitcompMode_t mode, bitcompAlgorithm_t algo)
  {
    bitcompResult_t result = bitcompCreatePlan(&handle_, n, dataType, mode, algo);
    if (result == BITCOMP_SUCCESS)
    {
      return;
    }
    else if (result == BITCOMP_INVALID_ALIGNMENT)
    {
      throw std::runtime_error("error:BITCOMP_INVALID_ALIGNMENT");
    }
    else if (result == BITCOMP_INVALID_PARAMETER)
    {
      throw std::runtime_error("error:BITCOMP_INVALID_PARAMETER");
    }
    else if (result == BITCOMP_INVALID_COMPRESSED_DATA)
    {
      throw std::runtime_error("error:BITCOMP_INVALID_COMPRESSED_DATA");
    }
    else if (result == BITCOMP_UNKNOWN_ERROR)
    {
      throw std::runtime_error("error:BITCOMP_UNKNOWN_ERROR");
    }
  }
  ~bitcompPlan() { bitcompDestroyPlan(handle_); }
  bitcompHandle_t *handleptr() { return &handle_; }
  bitcompHandle_t handle_;
};

/**
 * @brief Class for a Bitcomp compressor.
 *
 * This class provides a Bitcomp compressor implementation that compresses and decompresses data using the Bitcomp library.
 * It is a subclass of the generic compressor class.
 *
 * @tparam decompressType The type of the decompressed data.
 * @tparam compressedType The type of the compressed data.
 */
template <typename decompressType, typename compressedType>
class compressor_bitcomp final
    : public compressor<decompressType, compressedType>
{
public:
  /**
   * @brief Constructor for the Bitcomp compressor.
   *
   * Initializes the Bitcomp compressor with the specified dimensions and compression parameters.
   *
   * @param n1 The size of the first dimension.
   * @param n2 The size of the second dimension.
   * @param n3 The size of the third dimension.
   * @param config_kind The configuration kind parameter.
                        0 -> MAX_FRACTION: Dynamically computes delta based on max fraction of the field. `range_fraction` parameter is required.
                        1 -> STD: Dynamically computes delta based on std deviation of the field. `nun_sigma` parameter is required.
                        2 -> STATIC_DELTA: Uses the defined delta for compression. `delta` parameter is required.
   * @param range_fraction The range fraction parameter. (Used only on config_kind=MAX_FRACTION).
   * @param num_sigma The number of sigma parameter.  (Used only on config_kind=STD).
   * @param delta The delta parameter.  (Used only on config_kind=STATIC_DELTA).
   * @param float_kind The kind of floating-point data ('float' or 'double').
   * @param algo The algorithm parameter. Acceptable values: "default" or "sparse".
   */
  explicit compressor_bitcomp(const std::size_t n1, const std::size_t n2,
                              const std::size_t n3,
                              const int config_kind = 0,
                              const double range_fraction = 0.0,
                              const double num_sigma = 0.0,
                              const double delta = 0.0,
                              const std::string &float_kind = "float",
                              const std::string &algo = "default")
  {

    if (config_kind == 0)
    {
      if (range_fraction <= 0.0)
      {
        throw std::invalid_argument("range_fraction should be greater than zero");
      }
    }
    else if (config_kind == 1)
    {
      if (num_sigma <= 0.0)
      {
        throw std::invalid_argument("num_sigma should be greater than zero");
      }
      if (range_fraction <= 0.0)
      {
        throw std::invalid_argument("range_fraction should be greater than zero");
      }
    }
    else if (config_kind == 2)
    {
      if (delta <= 0.0)
      {
        throw std::invalid_argument("delta should be greater than zero");
      }
    }
    else
    {
      throw std::invalid_argument("Accepted config_kind: 0 {range_fraction}, 1 {range_Fraction, num_sigma}, 2 {delta}");
    }

    kind_ = config_kind;
    num_sigma_ = num_sigma;
    range_fraction_ = range_fraction;
    delta_ = delta;
    plan_ = buildPlan(n1, n2, n3, float_kind, algo);
  }

protected:
  size_t compress(decompressType *buf_in, compressedType *buf_out) override
  {
    bitcompDataType_t data_type;
    std::size_t size;
    check_return_nvcomp_bitcomp(
        bitcompGetDataTypeFromHandle(*plan_->handleptr(), &data_type));
    check_return_nvcomp_bitcomp(
        bitcompGetUncompressedSizeFromHandle(*plan_->handleptr(), &size));
    double delta = 0.0;
    if (kind_ == 0)
    {
      if (data_type == BITCOMP_FP32_DATA)
      {
        delta = max_fraction_ * maxFloat(reinterpret_cast<float *>(buf_in), size / sizeof(float));
      }
      else
      {
        delta = max_fraction_ * maxDouble(reinterpret_cast<double *>(buf_in),
                                          size / sizeof(double));
      }
    }
    else if (kind_ == 1)
    {
      if (data_type == BITCOMP_FP32_DATA)
      {
        const auto pair = meanStdFloat(reinterpret_cast<float *>(buf_in), size / sizeof(float));
        const auto range = ((pair.first + num_sigma_ * pair.second) -
                            (pair.first - num_sigma_ * pair.second));
        delta = range / range_fraction_;
      }
      else
      {
        const auto pair = meanStdDouble(reinterpret_cast<double *>(buf_in),
                                        size / sizeof(double));
        const auto range = ((pair.first + num_sigma_ * pair.second) -
                            (pair.first - num_sigma_ * pair.second));
        delta = range * range_fraction_;
      }
    }
    else if (kind_ == 2)
    {
      delta = delta_;
    }
    else
    {
      throw std::runtime_error("invalid value for kind_");
    }

    if (data_type == BITCOMP_FP32_DATA)
    {
      check_return_nvcomp_bitcomp(bitcompCompressLossy_fp32(
          *plan_->handleptr(), reinterpret_cast<float *>(buf_in),
          reinterpret_cast<void *>(buf_out), delta));
    }
    else if (data_type == BITCOMP_FP64_DATA)
    {
      check_return_nvcomp_bitcomp(bitcompCompressLossy_fp64(
          *plan_->handleptr(), reinterpret_cast<double *>(buf_in),
          reinterpret_cast<void *>(buf_out), delta));
    }
    else
    {
      throw std::runtime_error("Invalid value for dataype");
    }

    return compressed_buffer_size(buf_out);
  }

  void decompress(compressedType *buf_in, decompressType *buf_out, std::size_t compressed_size = -1) override
  {
    check_return_nvcomp_bitcomp(
        bitcompUncompress(*plan_->handleptr(), reinterpret_cast<void *>(buf_in),
                          reinterpret_cast<void *>(buf_out)));
  }
  std::string name() override { return "nvcomp-bitcomp-compressor"; }
  std::string description() override
  {
    return "nvcomp-bitcomp-compressor descr";
  }
  std::size_t compressed_buffer_size(compressedType *buf) override
  {
    std::size_t size;
    check_return_nvcomp_bitcomp(bitcompGetCompressedSize(buf, &size));
    return size;
  }
  std::size_t compressed_buffer_max_size() override
  {
    std::size_t size;
    check_return_nvcomp_bitcomp(
        bitcompGetUncompressedSizeFromHandle(*plan_->handleptr(), &size));
    return bitcompMaxBuflen(size);
  }

private:
  void check_return_nvcomp_bitcomp(bitcompResult_t result)
  {
    if (result == BITCOMP_SUCCESS)
    {
      return;
    }
    else if (result == BITCOMP_INVALID_ALIGNMENT)
    {
      throw std::runtime_error("error:BITCOMP_INVALID_ALIGNMENT");
    }
    else if (result == BITCOMP_INVALID_PARAMETER)
    {
      throw std::runtime_error("error:BITCOMP_INVALID_PARAMETER");
    }
    else if (result == BITCOMP_INVALID_COMPRESSED_DATA)
    {
      throw std::runtime_error("error:BITCOMP_INVALID_COMPRESSED_DATA");
    }
    else if (result == BITCOMP_UNKNOWN_ERROR)
    {
      throw std::runtime_error("error:BITCOMP_UNKNOWN_ERROR");
    }
  }

  std::unique_ptr<bitcompPlan>
  buildPlan(const std::size_t n1, const std::size_t n2, const std::size_t n3,
            const std::string &float_kind, const std::string &algo)
  {
    if (!(n1 * n2 * n3 > 0))
    {
      throw std::invalid_argument("n1*n2*n3 must be greater than zero");
    }
    bitcompDataType_t dataType;
    bitcompAlgorithm_t algo_;
    std::size_t elem_size;
    if (float_kind == "float")
    {
      dataType = BITCOMP_FP32_DATA;
      elem_size = sizeof(float);
    }
    else if (float_kind == "double")
    {
      dataType = BITCOMP_FP64_DATA;
      elem_size = sizeof(double);
    }
    else
    {
      throw std::invalid_argument("invalid argument for float_kind");
    }
    if (algo == "default")
    {
      algo_ = BITCOMP_DEFAULT_ALGO;
    }
    else if (algo == "sparse")
    {
      algo_ = BITCOMP_SPARSE_ALGO;
    }
    else
    {
      throw std::invalid_argument("invalid argument for algo");
    }
    return std::make_unique<bitcompPlan>((n1 * n2 * n3) * elem_size, dataType,
                                         BITCOMP_LOSSY_FP_TO_SIGNED, algo_);
  }
  double max_fraction_;
  double num_sigma_, range_fraction_, delta_;
  int kind_;
  std::unique_ptr<bitcompPlan> plan_;
};
