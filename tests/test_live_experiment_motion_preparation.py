"""Exact preparation changes must leave the CU motion inputs unchanged."""
import cv2
import numpy as np
import pytest

from fdm._native import require_native
from fdm.live_experiment.motion import dark_centroid, gray_spectrum


def centroid_reference(gray):
    weights = 255 - gray
    columns = weights.sum(axis=0, dtype=np.int64)
    rows = weights.sum(axis=1, dtype=np.int64)
    total = int(columns.sum())
    if total == 0:
        return (-1, -1)
    return (int(columns @ np.arange(gray.shape[1], dtype=np.int64)) // total,
            int(rows @ np.arange(gray.shape[0], dtype=np.int64)) // total)


@pytest.mark.parametrize('shape', [(1, 1), (1, 257), (193, 1), (127, 257), (910, 1365)])
@pytest.mark.parametrize('transform', ['identity', 'flip', 'slice', 'transpose'])
def test_centroid_exact_integer_moments_on_borrowed_inputs(shape, transform):
    gray = np.random.default_rng(20261015).integers(0, 256, shape, np.uint8)
    gray = {'identity': gray, 'flip': gray[::-1, ::-1], 'slice': gray[::2, ::2],
            'transpose': gray.T}[transform]
    gray.setflags(write=False)
    before = gray.copy()
    assert dark_centroid(gray) == centroid_reference(gray)
    np.testing.assert_array_equal(gray, before)


@pytest.mark.parametrize('shade', [0, 1, 127, 254, 255])
def test_centroid_constant_and_zero_mass(shade):
    gray = np.full((128, 256), shade, np.uint8)
    assert dark_centroid(gray) == centroid_reference(gray)


@pytest.mark.parametrize('shape', [(1, 1), (17, 31), (910, 1365)])
def test_float32_lookup_and_complete_spectrum_are_bitwise_equal(shape):
    gray = np.random.default_rng(35).integers(0, 256, shape, np.uint8)
    for current in (gray, gray[::-1, ::-1], gray[:, ::2]):
        reference = cv2.dft((current.astype(np.float64) / 255.0).astype(np.float32),
                            flags=cv2.DFT_COMPLEX_OUTPUT)
        np.testing.assert_array_equal(gray_spectrum(current), reference)


@pytest.mark.parametrize('gray,error', [(np.zeros((2, 3), np.float32), TypeError),
    (np.zeros((0, 2), np.uint8), ValueError), (np.zeros((2, 2, 1), np.uint8), ValueError)])
def test_centroid_rejects_invalid_native_input_and_recovers(gray, error):
    with pytest.raises(error):
        require_native().cu_motion_centroid(gray)
    assert dark_centroid(np.zeros((3, 5), np.uint8)) == (2, 1)


@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('shape', [(2160, 3840), (2730, 4096), (2049, 2049)])
@pytest.mark.parametrize('view', ['normal', 'flipped', 'transposed'])
def test_large_color_blocks_preserve_whole_frame_opencv_gray_resize(shape, channels, view):
    from fdm.live_experiment.motion import reduced_gray
    pixels = np.random.default_rng(612).integers(0, 256, (*shape, channels), np.uint8)
    if view == 'flipped':
        pixels = pixels[::-1, ::-1]
    elif view == 'transposed':
        pixels = pixels.transpose(1, 0, 2)
    pixels.flags.writeable = False
    code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(pixels, code)
    expected = cv2.resize(gray, (pixels.shape[1] // 3, pixels.shape[0] // 3), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(reduced_gray(pixels), expected)


def test_contiguous_large_gray_conversion_borrows_the_frame(monkeypatch):
    from fdm.live_experiment.motion import reduced_gray
    pixels = np.zeros((2049, 2049, 3), np.uint8)
    pixels.flags.writeable = False
    convert = cv2.cvtColor
    borrowed = []

    def inspect(source, code, *args, **kwargs):
        borrowed.append(np.shares_memory(source, pixels) and not source.flags.writeable)
        return convert(source, code, *args, **kwargs)

    monkeypatch.setattr(cv2, 'cvtColor', inspect)
    assert reduced_gray(pixels).shape == (683, 683)
    assert borrowed and all(borrowed)
