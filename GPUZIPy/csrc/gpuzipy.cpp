/**
 * @file gpuzipy.cpp
 * @brief A python wrapper for GPUZIP
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @date Feb 24, 2024
 */
#include <pybind11/pybind11.h>
#include "../../src/Compressor/include/compressor_zfp.hpp"
#include "../../src/Compressor/include/compressor_nvcomp_bitcomp.hpp"
#include "../../src/Compressor/include/compressor_cuszp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/**
 * @brief Compress data using the specified compressor.
 *
 * @param compressor The compressor object.
 * @param d_uncompressed_ptr Pointer to the uncompressed data (GPU memory pointer).
 * @param d_compressed_ptr Pointer to store the compressed data (GPU memory pointer).
 *                         The allocated memory needs to meet the minimum size of compressed_buffer_size.
 *                         If you don't previously know the field to be compressed, you can use compressed_buffer_size.
 */
std::size_t compress(compressor<void, void> *compressor, long d_uncompressed_ptr, long d_compressed_ptr)
{
     void *d_uncompressed = reinterpret_cast<void *>(d_uncompressed_ptr);
     void *d_compressed = reinterpret_cast<void *>(d_compressed_ptr);
     return compressor->Compress(d_uncompressed, d_compressed);
}

/**
 * @brief Decompress data using the specified compressor.
 *
 * @param compressor The compressor object.
 * @param d_compressed_ptr Pointer to the compressed data (GPU memory pointer).
 * @param d_uncompressed_ptr Pointer to store the decompressed data (GPU memory pointer).
 */
void decompress(compressor<void, void> *compressor, long d_compressed_ptr, long d_uncompressed_ptr, std::size_t compressed_size = -1)
{
     void *d_compressed = reinterpret_cast<void *>(d_compressed_ptr);
     void *d_uncompressed = reinterpret_cast<void *>(d_uncompressed_ptr);
     compressor->Decompress(d_compressed, d_uncompressed, compressed_size);
}

/**
 * @brief Get the maximum buffer size for compressed data. Gives a pessimistic size required for the compressed data.
 *
 * @param compressor The compressor object.
 * @return std::size_t The maximum buffer size.
 */
std::size_t compressed_buffer_max_size(compressor<void, void> *compressor)
{
     return compressor->Compressed_buffer_max_size();
}

/**
 * @brief Get the actual compressed buffer size in bytes.
 *
 * @param compressor The compressor object.
 * @param d_uncompressed_ptr Pointer to the uncompressed data.
 * @return std::size_t The compressed buffer size.
 */
std::size_t compressed_buffer_size(compressor<void, void> *compressor, long d_uncompressed_ptr)
{
     void *d_uncompressed = reinterpret_cast<void *>(d_uncompressed_ptr);
     return compressor->Compressed_buffer_size(d_uncompressed);
}

PYBIND11_MODULE(gpuzipy, m)
{
     m.doc() = "GPUZIP Compressors."; // Module level documentation

     py::class_<compressor<void, void>>(m, "Compressor");

     py::class_<compressor_zfp<void, void>, compressor<void, void>>(m, "CompressorZFP")
         .def(py::init<std::size_t, std::size_t, std::size_t, const std::string &, double>(),
              "Initialize CompressorZFP object",
              py::arg("n1"), py::arg("n2"), py::arg("n3"), py::arg("float_kind"), py::arg("rate"));

     py::class_<compressor_cuszp<void, void>, compressor<void, void>>(m, "CompressorCuszp")
         .def(py::init<std::size_t, std::size_t, std::size_t, double>(),
              "Initialize CompressorCuszp object",
              py::arg("n1"), py::arg("n2"), py::arg("n3"), py::arg("error_bound"));

     py::class_<compressor_bitcomp<void, void>, compressor<void, void>>(m, "CompressorBitcomp")
         .def(py::init<std::size_t, std::size_t, std::size_t,
                       const int, const double, const double, const double,
                       const std::string &, const std::string &>(),
              R"pbdoc(
               Initialize CompressorBitcomp object.

               Args:
                    n1 (int): Size of dimension 1.
                    n2 (int): Size of dimension 2.
                    n3 (int): Size of dimension 3.
                    config_kind (int): Configuration kind. 
                                        0 -> MAX_FRACTION: Dynamically computes delta based on max fraction of the field. `range_fraction` parameter is required.
                                        1 -> STD: Dynamically computes delta based on std deviation of the field. `nun_sigma` parameter is required.
                                        2 -> STATIC_DELTA: Uses the defined delta for compression. `delta` parameter is required.
                    range_fraction (float): Range fraction. (Used only on config_kind=MAX_FRACTION).
                    num_sigma (float): Number of sigma. (Used only on config_kind=STD).
                    delta (float): Delta. (Used only on config_kind=STATIC_DELTA).
                    float_kind (str): Float kind. Acceptable values: "float" or "double".
                    algo (str): Algorithm name. Acceptable values: "default" or "sparse".
               )pbdoc",
              py::arg("n1"), py::arg("n2"), py::arg("n3"),
              py::arg("config_kind"), py::arg("range_fraction"), py::arg("num_sigma"),
              py::arg("delta"), py::arg("float_kind"), py::arg("algo"));

     m.def("compressed_buffer_max_size", &compressed_buffer_max_size, R"pbdoc(
          Get the maximum buffer size for compressed data. Gives a pessimistic size required for the compressed data.
     )pbdoc");

     m.def("compressed_buffer_size", &compressed_buffer_size, R"pbdoc(
          Returns the exact compressed size in bytes.
     )pbdoc");

     m.def("compress", &compress, R"pbdoc(
          Compress
     )pbdoc");

     m.def("decompress", &decompress, R"pbdoc(
          Decompress
     )pbdoc");

#ifdef VERSION_INFO
     m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
     m.attr("__version__") = "dev";
#endif
}
