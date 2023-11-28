#pragma once
#include <array>
#include <exception>
#include <stdexcept>
#include <memory>
#include <zfp.h>

#include "compressor.hpp"

class field3d
{
public:
  explicit field3d(const std::size_t n1, const std::size_t n2,
                   const std::size_t n3, const std::string &float_kind)
  {
    zfp_type type;
    if (float_kind == "float")
    {
      type = zfp_type_float;
    }
    else if (float_kind == "double")
    {
      type = zfp_type_double;
    }
    else
    {
      throw std::invalid_argument("float_kind should be double or float");
    }
    field_ = zfp_field_3d(nullptr, type, n1, n2, n3);
  }
  field3d(const field3d &other) { field_ = remake(other.field_); }
  field3d &operator=(const field3d &other)
  {
    if (this == &other)
    {
      return *this;
    }
    if (field_)
    {
      zfp_field_free(field_);
    }
    field_ = remake(other.field_);
    return *this;
  }
  ~field3d() { zfp_field_free(field_); }
  void set_buffer(void *ptr) { zfp_field_set_pointer(field_, ptr); }

  void *get_buffer() { return zfp_field_pointer(field_); }
  std::string gettypestring()
  {
    if (zfp_field_type(field_) == zfp_type_float)
    {
      return "float";
    }
    else
    {
      return "double";
    }
  }
  zfp_type gettype() { return zfp_field_type(field_); }
  unsigned int getdims() { return zfp_field_dimensionality(field_); }

  void *get_field_ptr() { return reinterpret_cast<void *>(field_); }
  void *get_buffer() const { return zfp_field_pointer(field_); }
  std::string gettypestring() const
  {
    if (zfp_field_type(field_) == zfp_type_float)
    {
      return "float";
    }
    else
    {
      return "double";
    }
  }
  zfp_type gettype() const { return zfp_field_type(field_); }
  unsigned int getdims() const { return zfp_field_dimensionality(field_); }
  void *get_field_ptr() const { return reinterpret_cast<void *>(field_); }

private:
  zfp_field *remake(zfp_field *f)
  {
    std::array<std::size_t, 3> shape;
    zfp_field_size(f, &shape[0]);
    zfp_field *ret = zfp_field_3d(zfp_field_pointer(f), zfp_field_type(f),
                                  shape[0], shape[1], shape[2]);
    return ret;
  }
  zfp_field *field_;
};

class zfpStream
{
public:
  explicit zfpStream(const double rate, const field3d &field)
  {
    zfp_ = zfp_stream_open(nullptr);
    zfp_stream_set_rate(zfp_, rate, field.gettype(), field.getdims(),
                        zfp_false);
  }
  ~zfpStream() { zfp_stream_close(zfp_); }
  zfpStream(const zfpStream &other) { zfp_ = remake(other.zfp_); }
  zfpStream &operator=(const zfpStream &other)
  {
    if (this == &other)
    {
      return *this;
    }
    if (zfp_)
    {
      zfp_stream_close(zfp_);
    }
    zfp_ = remake(other.zfp_);
    return *this;
  }
  std::size_t buffersize(const field3d &field)
  {
    auto *p = reinterpret_cast<zfp_field *>(field.get_field_ptr());
    return zfp_stream_maximum_size(zfp_, p);
  }
  void *getStream() { return reinterpret_cast<void *>(zfp_); }

private:
  zfp_stream *remake(zfp_stream *p)
  {
    zfp_stream *ret = zfp_stream_open(nullptr);
    auto mode = zfp_stream_mode(p);
    zfp_stream_set_mode(ret, mode);
    return ret;
  }
  zfp_stream *zfp_;
};

/**
 * @brief Class for a ZFP compressor.
 * 
 * This class provides a ZFP compressor implementation that compresses and decompresses data using the ZFP library.
 * It is a subclass of the generic compressor class.
 * 
 * @tparam decompressType The type of the decompressed data.
 * @tparam compressedType The type of the compressed data.
 */
template <typename decompressType, typename compressedType>
class compressor_zfp final : public compressor<decompressType, compressedType>
{
public:
  /**
   * @brief Constructor for the ZFP compressor.
   *
   * Initializes the ZFP compressor with the specified dimensions and compression parameters.
   *
   * @param n1 The size of the first dimension.
   * @param n2 The size of the second dimension.
   * @param n3 The size of the third dimension.
   * @param float_kind The kind of floating-point data ('float' or 'double').
   * @param rate The compression rate parameter - See https://zfp.readthedocs.io/en/release0.5.4/modes.html#mode-fixed-rate
   */
  explicit compressor_zfp(const std::size_t n1, const std::size_t n2,
                          const std::size_t n3, const std::string float_kind,
                          const double rate)
      : field_(n1, n2, n3, float_kind), stream_(rate, field_) {}

protected:
  size_t compress(decompressType *buf_in, compressedType *buf_out) override
  {
    field_.set_buffer(reinterpret_cast<void *>(buf_in));
    auto *zfpField = reinterpret_cast<zfp_field *>(field_.get_field_ptr());
    auto *zfpStream = reinterpret_cast<zfp_stream *>(stream_.getStream());
    auto *bitStream = stream_open(reinterpret_cast<void *>(buf_out),
                                  stream_.buffersize(field_));

    zfp_stream_set_bit_stream(zfpStream, bitStream);
    zfp_stream_rewind(zfpStream);
    zfp_stream_set_execution(zfpStream, zfp_exec_cuda);
    zfp_compress(zfpStream, zfpField);
    stream_close(bitStream);

    return compressed_buffer_size(buf_out);
  }
  void decompress(compressedType *buf_in, decompressType *buf_out, size_t compressed_size = -1) override
  {
    field_.set_buffer(reinterpret_cast<void *>(buf_out));
    auto *zfpField = reinterpret_cast<zfp_field *>(field_.get_field_ptr());
    auto *zfpStream = reinterpret_cast<zfp_stream *>(stream_.getStream());
    auto *bitStream = stream_open(reinterpret_cast<void *>(buf_in),
                                  stream_.buffersize(field_));
    zfp_stream_set_bit_stream(zfpStream, bitStream);
    zfp_stream_rewind(zfpStream);
    zfp_stream_set_execution(zfpStream, zfp_exec_cuda);
    zfp_decompress(zfpStream, zfpField);
    stream_close(bitStream);
  }
  std::string name() override { return "zfp compressor"; }
  std::string description() override { return "zfp compressor descr"; }
  std::size_t compressed_buffer_size(compressedType *buf = nullptr) override
  {
    return stream_.buffersize(field_);
  }
  std::size_t compressed_buffer_max_size() override
  {
    return stream_.buffersize(field_);
  }

private:
  field3d field_;
  zfpStream stream_;
};
