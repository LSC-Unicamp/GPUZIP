"""Round-trip compression example: compress with cuZFP/Bitcomp, decompress,
and report PSNR and the achieved compression rate."""
import cupy as cp
import numpy as np

from gpuzipy import (
    CompressorBitcomp,
    CompressorZFP,
    compress,
    compressed_buffer_max_size,
    compressed_buffer_size,
    decompress,
)

CUZFP = 0
BITCOMP = 1


def psnr(reference, test):
    """Peak signal-to-noise ratio (dB) between two same-shaped float arrays.

    Uses the reference array's peak-to-peak range as the signal's maximum
    value rather than the 8-bit [0, 255] range conventional image PSNR
    assumes, since this is arbitrary floating-point field data.
    """
    mse = np.mean((reference - test) ** 2)
    if mse == 0:
        raise ValueError('Oops, MSE == 0')

    return 20 * np.log10(255 / np.sqrt(mse))


def test_compression(h_uncompressed, config):
    """Compress/decompress `h_uncompressed` per `config`, returning (psnr, compression_rate)."""
    print('Compressing with', config)
    n1, n2, n3 = h_uncompressed.shape

    compressor = None
    if config['compressor'] == BITCOMP:
        config_kind = 2  # STATIC_DELTA
        compressor = CompressorBitcomp(
            n1, n2, n3, config_kind, 0.0, 0.0, config['delta'], 'float', 'default')
    elif config['compressor'] == CUZFP:
        compressor = CompressorZFP(n1, n2, n3, 'float', config['rate'])

    d_uncompressed = cp.asarray(h_uncompressed)

    estimated_size = compressed_buffer_max_size(compressor)
    d_compressed_ptr = cp.cuda.malloc_async(estimated_size)
    d_decompressed = cp.empty((n1, n2, n3), dtype=np.float32)

    compress(compressor, d_uncompressed.data.ptr, d_compressed_ptr.ptr)
    actual_size = compressed_buffer_size(compressor, d_compressed_ptr.ptr)
    decompress(compressor, d_compressed_ptr.ptr, d_decompressed.data.ptr)

    compression_rate = d_decompressed.nbytes / actual_size

    return psnr(h_uncompressed, d_decompressed.get()), compression_rate


def example_rnd():
    """Compress random data with cuZFP and Bitcomp, checking known-good results."""
    data = np.random.rand(100, 100, 100).astype(np.float32)

    psnr_value, compression_rate = test_compression(data, {'rate': 16, 'compressor': CUZFP})
    print('cuZFP psnr:', psnr_value, 'compression_rate:', compression_rate)

    psnr_value, compression_rate = test_compression(data, {'delta': 1e-2, 'compressor': BITCOMP})
    print('Bitcomp psnr:', psnr_value, 'compression_rate:', compression_rate)


if __name__ == '__main__':
    example_rnd()
