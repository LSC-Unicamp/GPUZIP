#pragma once
#include <memory>
#include <nvtx3/nvToolsExt.h>
#include <string>

/**
 * @brief Template class for a compressor.
 *
 * This class provides a template for compressors, which are used to compress and decompress data.
 * The compressor supports profiling to measure performance.
 *
 * @tparam decompressType The type of the decompressed data - "void" is a good choice ;-)
 * @tparam compressedType The type of the compressed data - "void" is a good choice ;-)
 * 
 * @author Alan Souza <alan.geof.ba@gmail.com>
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 *
 * @date Feb 5, 2023
 */
template <typename decompressType, typename compressedType>
class Compressor
{
public:
  /**
   * @brief Constructor for the compressor class.
   *
   * Initializes the compressor with profiling disabled by default.
   */
  Compressor() : profile_{false} {}
  
  virtual ~Compressor() = default;

  /**
   * @brief Compress data.
   *
   * Compresses the input data buffer and writes the compressed data to the output buffer.
   * If profiling is enabled, this operation is wrapped with an NVTX range for profiling.
   *
   * @param buf_in Pointer to the input data buffer.
   * @param buf_out Pointer to the output buffer to store compressed data.
   */
  std::size_t Compress(decompressType *buf_in, compressedType *buf_out)
  {
    if (profile_)
    {
      nvtxRangePush(cname_.c_str());
    }

    std::size_t compressed_size = compress(buf_in, buf_out);

    if (profile_)
    {
      nvtxRangePop();
    }

    return compressed_size;
  }

  /**
   * @brief Decompress data.
   *
   * Decompresses the input data buffer and writes the decompressed data to the output buffer.
   * If profiling is enabled, this operation is wrapped with an NVTX range for profiling.
   *
   * @param buf_in Pointer to the input data buffer containing compressed data.
   * @param buf_out Pointer to the output buffer to store decompressed data.
   */
  void Decompress(compressedType *buf_in, decompressType *buf_out, std::size_t compressed_size = -1)
  {
    if (profile_)
    {
      nvtxRangePush(dname_.c_str());
    }
    decompress(buf_in, buf_out, compressed_size);
    if (profile_)
    {
      nvtxRangePop();
    }
  }

  /**
   * @brief Get the name of the compressor - Useful for debugging and logging.
   *
   * @return The name of the compressor.
   */
  std::string Name() { return name(); }

  /**
   * @brief Get the description of the compressor - Useful for debugging and logging.
   *
   * @return The description of the compressor.
   */
  std::string Description() { return description(); }

  /**
   * @brief Get the size of the compressed data buffer.
   *
   * @param buf Pointer to the compressed data buffer. If null, returns the maximum size.
   * @return The size of the compressed data buffer.
   */
  std::size_t CompressedBufferSize(compressedType *buf = nullptr)
  {
    return compressedBufferSize(buf);
  }

  /**
   * @brief Get the maximum size of the compressed data buffer.
   *
   * @return The maximum size of the compressed data buffer.
   */
  std::size_t CompressedMaxSize()
  {
    return compressedMaxSize();
  }

  /**
   * @brief Enable profiling for the compressor.
   *
   * Generates domain names and marks the start of profiling for the compressor.
   */
  void EnableProfile()
  {
    std::string domainName = "compressor-domain-" + Name();
    cname_ = "Compressor-" + Name();
    dname_ = "Decompressor-" + Name();
    std::ptrdiff_t addr = reinterpret_cast<std::ptrdiff_t>(this);
    std::string mark =
        "Enable profile for compressor, description:" + Description() +
        "ptr:" + std::to_string(addr);
    nvtxMarkA(mark.c_str());
    profile_ = true;
  }

  /**
   * @brief Disable profiling for the compressor.
   *
   * Disables profiling for the compressor.
   */
  void DisableProfile() { profile_ = false; }

protected:
  /**
   * @brief Compress data.
   *
   * This method should be implemented by subclasses to perform the actual compression operation.
   *
   * @param buf_in Pointer to the input data buffer.
   * @param buf_out Pointer to the output buffer to store compressed data.
   */
  virtual std::size_t compress(decompressType *buf_in, compressedType *buf_out) = 0;

  /**
   * @brief Decompress data.
   *
   * This method should be implemented by subclasses to perform the actual decompression operation.
   *
   * @param buf_in Pointer to the input data buffer containing compressed data.
   * @param buf_out Pointer to the output buffer to store decompressed data.
   */
  virtual void decompress(compressedType *buf_in, decompressType *buf_out, std::size_t compressed_size = -1) = 0;

  /**
   * @brief Get the name of the compressor.
   *
   * This method should be implemented by subclasses to provide the name of the compressor.
   *
   * @return The name of the compressor.
   */
  virtual std::string name() = 0;

  /**
   * @brief Get the description of the compressor.
   *
   * This method should be implemented by subclasses to provide a description of the compressor.
   *
   * @return The description of the compressor.
   */
  virtual std::string description() = 0;

  /**
   * @brief Get the size of the compressed data buffer.
   *
   * This method should be implemented by subclasses to provide the size of the compressed data buffer.
   *
   * @param buf Pointer to the compressed data buffer. If null, returns the maximum size.
   * @return The size of the compressed data buffer.
   */
  virtual std::size_t compressedBufferSize(compressedType *buf) = 0;

  /**
   * @brief Get the maximum size of the compressed data buffer.
   *
   * This method should be implemented by subclasses to provide the maximum size of the compressed data buffer.
   *
   * @return The maximum size of the compressed data buffer.
   */
  virtual std::size_t compressedMaxSize() = 0;

private:
  std::string cname_, dname_;
  bool profile_;
};
