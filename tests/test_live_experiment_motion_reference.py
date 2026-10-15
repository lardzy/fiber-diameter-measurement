"""Mathematical golden observations plus source-lifecycle motion regressions."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.validation.validate_cu_motion_reference import replay
from fdm.live_experiment.frames import FrameHandle
from fdm.live_experiment.motion import (
    FieldMotionDetector, compare_geometry, correlation_shift, dark_centroid,
    gray_spectrum, reduced_gray,
)


@pytest.mark.parametrize("record", replay(), ids=lambda r: f"{r['product']}-{r['group']}-{r['input']}")
def test_recorded_motion_helpers(record):
    assert record["passed"], record


def test_unsigned_fix_changes_only_documented_strong_sobel_decisions():
    changed = [row for row in replay() if row.get("corrected_differs")]
    assert len(changed) == 12
    assert all(row["legacy_stable"] and not row["corrected_stable"] for row in changed)


def test_color_is_reduced_after_gray_conversion():
    random = np.random.default_rng(521).integers(0, 256, (77, 101, 3), dtype=np.uint8)
    expected = cv2.resize(cv2.cvtColor(random, cv2.COLOR_BGR2GRAY), (33, 25), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(reduced_gray(random), expected)
    assert not np.array_equal(expected, cv2.cvtColor(cv2.resize(random, (33, 25)), cv2.COLOR_BGR2GRAY))


def test_fft_constant_tie_is_deterministic_and_does_not_spawn_threads():
    black = np.zeros((96, 128), np.uint8)
    gray = np.full((96, 128), 100, np.uint8)
    assert correlation_shift(gray_spectrum(black), black) == (0, 0)
    assert correlation_shift(gray_spectrum(gray), gray) == (1, 0)
    assert gray_spectrum(gray).dtype == np.float32
    assert dark_centroid(np.full_like(gray, 255)) == (-1, -1)


def test_half_width_correlation_shift_uses_strict_wrap_boundary():
    random = np.random.default_rng(3).integers(0, 256, (20, 60), np.uint8)
    assert correlation_shift(gray_spectrum(random), np.roll(random, 30, axis=1)) == (30, 0)
    assert correlation_shift(gray_spectrum(random), np.roll(random, 31, axis=1)) == (-29, 0)


def test_motion_references_are_owned_and_generation_change_resets():
    pixels = np.random.default_rng(2).integers(0, 256, (96, 128), np.uint8)
    detector = FieldMotionDetector()
    first = FrameHandle.snapshot(pixels, source_id="test", generation=1, sequence=1, live=True)
    detector.compare(first)
    previous = detector.reference.copy()
    pixels[:] = 0
    np.testing.assert_array_equal(detector.reference, previous)
    current = FrameHandle.snapshot(pixels, source_id="test", generation=2, sequence=1, live=True)
    assert detector.compare(current).reason == "reference"
    assert detector.compare(current).stable


@pytest.mark.parametrize("reference,current,error", [
    (np.zeros((1, 1), np.uint8), np.zeros((2, 2), np.uint8), ValueError),
    (np.zeros((0, 1), np.uint8), np.zeros((0, 1), np.uint8), ValueError),
    (np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32), TypeError),
])
def test_geometry_rejects_invalid_inputs(reference, current, error):
    with pytest.raises(error):
        compare_geometry(reference, current)


def test_spectrum_lookup_preserves_every_reference_float_value():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected = cv2.dft((gray.astype(np.float64) / 255.0).astype(np.float32),
                       flags=cv2.DFT_COMPLEX_OUTPUT)
    np.testing.assert_array_equal(gray_spectrum(gray), expected)


def test_exact_geometry_cache_keeps_sobel_and_invalidates_on_reference_change(monkeypatch):
    import fdm.live_experiment.motion as motion
    original = motion.compare_geometry
    calls = []

    def count(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(motion, "compare_geometry", count)
    detector = FieldMotionDetector()
    reference = np.random.default_rng(22).integers(0, 256, (96, 128), dtype=np.uint8)
    current = np.roll(reference, 1, axis=1)
    key = ("camera", 1, (384, 288))
    detector.set_reference_reduced(reference, key)
    first = detector.compare_reduced(current, key, "one", 1)
    repeat = detector.compare_reduced(current.copy(), key, "two", 2)
    assert len(calls) == 1  # Equal pixels in a different allocation are proven.
    assert first.reason == "appearance" and not first.stable
    assert repeat.reason == "stable" and repeat.changed_fraction == 0.0
    assert repeat.frame_id == "two" and repeat.sequence == 2
    changed = current.copy()
    changed[3, 5] ^= 1
    detector.compare_reduced(changed, key, "three", 3)
    assert len(calls) == 2  # Even one different pixel invalidates the cache.
    detector.set_reference_reduced(reference.copy(), key)
    detector.compare_reduced(current.copy(), key, "four", 4)
    assert len(calls) == 3
    detector.reset()
    assert detector.retained_bytes == 0


def test_geometry_cache_reuses_rejections_without_advancing_previous(monkeypatch):
    import fdm.live_experiment.motion as motion
    original = motion.compare_geometry
    calls = []

    def count(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(motion, "compare_geometry", count)
    detector = FieldMotionDetector()
    reference = np.random.default_rng(47).integers(0, 256, (96, 128), dtype=np.uint8)
    current = np.roll(reference, 26, axis=1)
    key = ("camera", 1, (384, 288))
    detector.set_reference_reduced(reference, key)
    for sequence in (1, 2):
        result = detector.compare_reduced(current.copy(), key, str(sequence), sequence)
        assert result.reason == "translation" and result.translation == (26, 0)
        assert detector.previous is reference
    assert len(calls) == 1
    assert detector.retained_bytes == reference.nbytes * 2 + detector.reference_fft.nbytes
