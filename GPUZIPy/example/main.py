"""Round-trip compression example: compress with cuZFP/Bitcomp, decompress,
and report PSNR and the achieved compression rate."""
import urllib.request
import zipfile
from pathlib import Path

import cupy as cp
import matplotlib
matplotlib.use('Agg')  # headless: this example only saves images, never shows a window
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

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

DATA_DIR = Path(__file__).parent / 'data'
# Marmousi3D velocity model, hosted on Unicamp's Dataverse (redu.unicamp.br). The server
# names it with a ".zipfile" extension, but the content is an ordinary zip archive.
MARMOUSI3D_URL = 'https://redu.unicamp.br/api/access/datafile/13547?gbrecs=true'
MARMOUSI3D_ARCHIVE = DATA_DIR / 'marmousi3d.bin.zipfile'
MARMOUSI3D_BIN = DATA_DIR / 'marmousi3d.bin'
MARMOUSI3D_N1, MARMOUSI3D_N2, MARMOUSI3D_N3 = 351, 901, 301
# SeisUtils' data2numpy convention: reshape as [n3, n2, n1], not [n1, n2, n3].
MARMOUSI3D_SHAPE = (MARMOUSI3D_N3, MARMOUSI3D_N2, MARMOUSI3D_N1)

OUTPUT_DIR = Path(__file__).parent / 'output'


def download_marmousi3d():
    """Download (if needed) and load the Marmousi3D velocity model as a float32 array."""
    DATA_DIR.mkdir(exist_ok=True)

    if not MARMOUSI3D_ARCHIVE.exists():
        print(f'Downloading {MARMOUSI3D_URL}')
        urllib.request.urlretrieve(MARMOUSI3D_URL, MARMOUSI3D_ARCHIVE)

    if not MARMOUSI3D_BIN.exists():
        with zipfile.ZipFile(MARMOUSI3D_ARCHIVE) as archive:
            archive.extract(MARMOUSI3D_BIN.name, DATA_DIR)

    return np.fromfile(MARMOUSI3D_BIN, dtype=np.float32).reshape(MARMOUSI3D_SHAPE)


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


def ssim(reference, test):
    """Average structural similarity (SSIM) across depth slices of two 3D arrays,
    matching SeisUtils' render/helpers.py _ssim convention."""
    data_range = reference.max() - reference.min()
    scores = [
        structural_similarity(reference[i], test[i], data_range=data_range)
        for i in range(reference.shape[0])
    ]
    return float(np.mean(scores))


def save_slice_image(frame, path, cmap='gray', colorbar=False, title=None):
    """Save a 2D array as an image, matching SeisUtils' render/helpers.py plot_image conventions."""
    plt.figure()
    plt.gca().invert_yaxis()
    im = plt.imshow(frame, cmap=cmap)
    if colorbar:
        plt.colorbar(im)
    if title:
        plt.title(title)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def render_comparison(name, original, decompressed, psnr_value, ssim_value):
    """Save an original/decompressed slice pair and their absolute error map
    to OUTPUT_DIR/name/, following SeisUtils' render/error_map.py convention."""
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # SeisUtils' error_map.py convention: index the frame along axis 0, then transpose
    # the 2D slice so depth ends up on the y-axis.
    mid_slice = original.shape[0] // 2
    original_slice = original[mid_slice].T
    decompressed_slice = decompressed[mid_slice].T
    error_slice = np.abs(original_slice - decompressed_slice)

    save_slice_image(original_slice, out_dir / 'original.png')
    save_slice_image(decompressed_slice, out_dir / 'decompressed.png')
    save_slice_image(
        error_slice, out_dir / 'error_map.png', cmap='viridis', colorbar=True,
        title=f'PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}')


def test_compression(h_uncompressed, config, name):
    """Compress/decompress `h_uncompressed` per `config`, returning (psnr, ssim, compression_rate)."""
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

    h_decompressed = d_decompressed.get()
    psnr_value = psnr(h_uncompressed, h_decompressed)
    ssim_value = ssim(h_uncompressed, h_decompressed)
    render_comparison(name, h_uncompressed, h_decompressed, psnr_value, ssim_value)

    return psnr_value, ssim_value, compression_rate


def example_marmousi3d():
    """Compress the Marmousi3D velocity model with cuZFP and Bitcomp."""
    data = download_marmousi3d()

    psnr_value, ssim_value, compression_rate = test_compression(
        data, {'rate': 16, 'compressor': CUZFP}, 'cuzfp')
    print('cuZFP psnr:', psnr_value, 'ssim:', ssim_value, 'compression_rate:', compression_rate)

    psnr_value, ssim_value, compression_rate = test_compression(
        data, {'delta': 1e-2, 'compressor': BITCOMP}, 'bitcomp')
    print('Bitcomp psnr:', psnr_value, 'ssim:', ssim_value, 'compression_rate:', compression_rate)


if __name__ == '__main__':
    example_marmousi3d()
