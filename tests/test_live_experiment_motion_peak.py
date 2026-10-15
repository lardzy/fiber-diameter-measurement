import numpy as np
import pytest

from fdm._native import require_native
from fdm.live_experiment.motion import correlation_peak_reference


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (2, 2), (3, 10), (20, 71), (73, 103)])
def test_native_peak_preserves_double_promotion_parity_ties_and_wrap(shape):
    rng = np.random.default_rng(20261015)
    arrays = [rng.uniform(-1e6, 1e6, (*shape, 2)).astype(np.float32),
              rng.integers(-2, 3, (*shape, 2)).astype(np.float32),
              np.zeros((*shape, 2), np.float32), np.ones((*shape, 2), np.float32)]
    for array in arrays:
        array.setflags(write=False)
        assert tuple(require_native().cu_motion_peak(array)) == correlation_peak_reference(array)


def test_peak_uses_imaginary_component_and_first_within_parity():
    array = np.zeros((8, 10, 2), np.float32)
    array[7, 8, 1] = 5
    array[6, 7, 0] = 5
    array[4, 9, 1] = -5
    assert tuple(require_native().cu_motion_peak(array)) == (-1, 4)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (1, 8), (2, 1), (3, 5), (4, 6)])
def test_single_pass_keeps_parity_order_for_each_possible_peak_position(shape):
    native = require_native()
    for y in range(shape[0]):
        for x in range(shape[1]):
            array = np.zeros((*shape, 2), np.float32)
            array[y, x, 0] = np.finfo(np.float32).max
            # A positive equal peak at the opposite end exercises first-within-
            # parity and odd-wins ties regardless of global row-major order.
            array[-1, -1, 1] = -np.finfo(np.float32).max
            assert tuple(native.cu_motion_peak(array)) == correlation_peak_reference(array)


@pytest.mark.parametrize("array,exception", [
    (np.zeros((2, 3, 2), np.float64), TypeError),
    (np.zeros((2, 3), np.float32), ValueError),
    (np.zeros((0, 3, 2), np.float32), ValueError),
    (np.zeros((2, 3, 1), np.float32), ValueError),
    (np.zeros((2, 3, 2), np.float32)[:, ::-1], ValueError),
    (np.full((2, 3, 2), np.nan, np.float32), ValueError),
    (np.full((2, 3, 2), np.inf, np.float32), ValueError),
])
def test_peak_rejects_invalid_buffers_and_can_be_called_again(array, exception):
    native = require_native()
    with pytest.raises(exception):
        native.cu_motion_peak(array)
    assert tuple(native.cu_motion_peak(np.zeros((2, 3, 2), np.float32))) == (0, 0)


@pytest.mark.parametrize("bits", [
    0x7F800000, 0xFF800000,  # Infinities.
    0x7FC00000, 0xFFC00000, 0x7FC00001, 0xFFC00001,  # Quiet NaNs.
    0x7FFFFFFF, 0xFFFFFFFF,  # Quiet NaNs with all payload bits set.
    0x7F800001, 0xFF800001, 0x7FBFFFFF, 0xFFBFFFFF,  # Signaling NaNs.
], ids=lambda bits: f"{bits:08x}")
@pytest.mark.parametrize("point", [(0, 0), (1, 2), (2, 4)])
@pytest.mark.parametrize("component", [0, 1])
def test_peak_rejects_every_nonfinite_encoding_without_mutation_and_recovers(bits, point, component):
    array = np.zeros((3, 5, 2), np.float32)
    # Assign through the integer view: Python float conversion would quiet sNaN.
    array.view(np.uint32)[*point, component] = bits
    original_bits = array.view(np.uint32).copy()
    array.setflags(write=False)
    native = require_native()
    with pytest.raises(ValueError, match="^correlation must contain finite values$"):
        native.cu_motion_peak(array)
    np.testing.assert_array_equal(array.view(np.uint32), original_bits)
    assert not array.flags.writeable
    assert tuple(native.cu_motion_peak(np.zeros((2, 3, 2), np.float32))) == (0, 0)


@pytest.mark.parametrize("bits", [
    0x00000000, 0x80000000,  # Signed zero.
    0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,  # Subnormals.
    0x00800000, 0x80800000,  # Smallest normal values.
    0x7F7FFFFF, 0xFF7FFFFF,  # Largest finite values.
], ids=lambda bits: f"{bits:08x}")
@pytest.mark.parametrize("component", [0, 1])
def test_peak_accepts_ieee_finite_boundaries_and_preserves_double_magnitude(bits, component):
    array = np.zeros((3, 5, 2), np.float32)
    array.view(np.uint32)[2, 4, component] = bits
    original_bits = array.view(np.uint32).copy()
    array.setflags(write=False)
    expected = (0, 0) if bits & 0x7FFFFFFF == 0 else (-1, -1)
    assert tuple(require_native().cu_motion_peak(array)) == expected
    assert correlation_peak_reference(array) == expected
    np.testing.assert_array_equal(array.view(np.uint32), original_bits)


@pytest.mark.parametrize("imaginary_exponent", [-13, -26, -27])
@pytest.mark.parametrize("stronger_parity", [0, 1])
def test_peak_near_ties_keep_double_rounding_and_odd_parity_tie_break(imaginary_exponent, stronger_parity):
    array = np.zeros((3, 6, 2), np.float32)
    array[0, :2, 0] = 1
    array[0, stronger_parity, 1] = np.float32(2.0 ** imaginary_exponent)
    # 2^-26 and 2^-52 survive double addition to 1, although not float32;
    # 2^-54 rounds away even in double, leaving the original odd-column tie.
    expected = (1 if imaginary_exponent == -27 else stronger_parity, 0)
    array.setflags(write=False)
    assert tuple(require_native().cu_motion_peak(array)) == expected
    assert correlation_peak_reference(array) == expected


def test_peak_rejects_unaligned_storage_and_recovers():
    array = np.ndarray((2, 3, 2), np.float32, buffer=bytearray(2 * 3 * 2 * 4 + 1), offset=1)
    native = require_native()
    with pytest.raises(ValueError, match="aligned storage"):
        native.cu_motion_peak(array)
    assert tuple(native.cu_motion_peak(np.zeros((2, 3, 2), np.float32))) == (0, 0)
