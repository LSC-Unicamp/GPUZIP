from gpuzipy import CompressorZFP, CompressorBitcomp, compressed_buffer_size, compressed_buffer_max_size, compress, decompress
import cupy as cp
import numpy as np
import math

CUZFP = 0
BITCOMP = 1
ERROR_BOUND = 2

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        raise ValueError('Oops, MSE == 0')

    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val

def test_compression(h_uncompressed, config):
    print('Compressing with', config)
    n1, n2, n3 = h_uncompressed.shape

    compressor = None
    if config['compressor'] == BITCOMP:
        ERROR_BOUND = 2
        ALGO_DEFAULT = 'default'
        ALGO_SPARSE = 'sparse'
        compressor =  CompressorBitcomp(n1, n2, n3, ERROR_BOUND, 0.0, 0.0, config['delta'], 'float', ALGO_DEFAULT)
    elif config['compressor'] == CUZFP:
        compressor =  CompressorZFP(n1, n2, n3, 'float', config['rate'])

    d_uncompressed = cp.asarray(h_uncompressed)

    estimated_size = compressed_buffer_max_size(compressor)

    d_compressed_ptr = cp.cuda.malloc_async(estimated_size)

    d_decompressed = cp.empty((n1, n2, n3), dtype=np.float32)

    compress(compressor, d_uncompressed.data.ptr, d_compressed_ptr.ptr)

    actual_size = compressed_buffer_size(compressor, d_compressed_ptr.ptr)

    decompress(compressor, d_compressed_ptr.ptr, d_decompressed.data.ptr)

    compression_rate = d_decompressed.nbytes/actual_size

    return (psnr(h_uncompressed, d_decompressed.get()), compression_rate,)



def example_rnd():
    data = np.random.rand(100, 100, 100).astype(np.float32)

    psnr, compression_rate = test_compression(data, { 'rate': 16, 'compressor': CUZFP})
    assert 52.9 == round(psnr, 1)
    assert 2 == round(compression_rate)

    psnr, compression_rate = test_compression(data, { 'delta': 1e8, 'compressor': BITCOMP})
    assert 52.9 == round(psnr, 1)
    assert 678 == round(compression_rate)


if __name__ == '__main__':
    example_rnd()

