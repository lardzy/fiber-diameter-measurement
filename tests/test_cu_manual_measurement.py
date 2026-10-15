"""Local geometry, ownership and cancellation checks for assisted measurement."""
from __future__ import annotations

import math
import sys
from threading import Event, Thread

import numpy as np
import pytest

from fdm import _native
from fdm.services.cu_manual_measurement import measure_manual


@pytest.fixture(scope="module")
def kernel():
    module = _native.require_native(validate_source=True)
    assert hasattr(module, "cu_manual_measure"), "rebuild native kernels for CU measurement"
    return module


@pytest.fixture
def strip():
    image = np.full((240, 320), 240, dtype=np.uint8)
    image[110:130, 20:300] = 30
    return image


def test_crossing_reports_fitted_width_not_input_length(kernel, strip):
    result = measure_manual(strip, (170, 85), (150, 155))
    assert result["status"] == "ok"
    assert result["width_px"] == 19.0
    assert len(result["edge_a"]) == len(result["edge_b"]) == 21
    assert math.dist(result["start"], result["end"]) > result["width_px"]
    assert result["diagnostics"]["algorithm_revision"] == "cu-manual-portable-v1"


def test_reversed_gesture_has_same_width_and_opposite_edges(kernel, strip):
    forward = measure_manual(strip, (160, 80), (160, 160))
    backward = measure_manual(strip, (160, 160), (160, 80))
    assert forward["width_px"] == backward["width_px"] == 19.0
    assert forward["start"] == backward["end"]
    assert forward["end"] == backward["start"]


@pytest.mark.parametrize("kind", ["gray", "bgr", "readonly", "flipped", "sliced", "channel_flip"])
def test_strides_and_input_immutability(kernel, strip, kind):
    if kind == "bgr":
        source = np.repeat(strip[..., None], 3, axis=2)
    elif kind == "readonly":
        source = strip.copy()
        source.setflags(write=False)
    elif kind == "flipped":
        source = strip[::-1, ::-1]
    elif kind == "sliced":
        wide = np.repeat(strip, 2, axis=1)
        source = wide[:, ::2]
    elif kind == "channel_flip":
        source = np.repeat(strip[..., None], 3, axis=2)[..., ::-1]
    else:
        source = strip
    original = source.copy()
    result = measure_manual(source, (160, 80), (160, 160))
    assert result["width_px"] == 19
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize(
    "start,end,reason",
    [((160, 50), (160, 51), "line_too_short"),
     ((160, 50), (160, 80), "no_complete_crossing"),
     ((160, 115), (160, 125), "endpoints_not_background"),
     ((100, 100), (230, 140), "crossing_too_oblique")],
)
def test_geometric_rejection_never_falls_back_to_free_distance(kernel, strip, start, end, reason):
    result = measure_manual(strip, start, end)
    assert result["status"] == "rejected"
    assert result["width_px"] is None
    assert result["reason"] == reason


def test_small_component_cleanup_keeps_true_fiber(kernel, strip):
    # An isolated 3x3 dark speck on the crossing must disappear before edge walk.
    strip[89:92, 159:162] = 30
    result = measure_manual(strip, (160, 80), (160, 160))
    assert result["width_px"] == 19
    assert result["diagnostics"]["removed_small_component_pixels"] == 9


def test_exact_36_pixel_component_is_rejected_and_37_is_kept(kernel):
    image = np.full((120, 120), 240, np.uint8)
    image[50:56, 50:56] = 30
    result = measure_manual(image, (52, 40), (52, 70))
    assert result["status"] == "rejected"
    assert result["diagnostics"]["removed_small_component_pixels"] == 36
    image[56, 52] = 30
    result = measure_manual(image, (52, 40), (52, 70))
    assert result["diagnostics"]["removed_small_component_pixels"] == 0


def test_boundary_adjustment_is_explicit(kernel, strip):
    result = measure_manual(strip, (160, 0), (160, 160))
    assert result["width_px"] == 19
    assert result["diagnostics"]["boundary_adjusted"] is True
    assert result["diagnostics"]["input_start"] == (160, 0)


def test_fractional_coordinates_use_original_pixel_truncation(kernel, strip):
    a = measure_manual(strip, (160.9, 80.9), (160.1, 160.1))
    b = measure_manual(strip, (160, 80), (160, 160))
    assert a == b


def test_6k_frame_uses_local_roi(kernel):
    source = np.full((4096, 6144), 240, np.uint8)
    source[2040:2060, 100:6000] = 30
    result = measure_manual(source, (3000, 2010), (3000, 2090))
    assert result["width_px"] == 19
    assert result["diagnostics"]["processed_pixels"] == 77 * 157


def test_enormous_gesture_is_rejected_before_allocating_scratch(kernel):
    source = np.broadcast_to(np.uint8(240), (20000, 20000))
    with pytest.raises(ValueError, match="预算"):
        measure_manual(source, (100, 100), (19900, 19900))


def test_cancellation_precedes_roi_budget_failure(kernel):
    source = np.broadcast_to(np.uint8(240), (20000, 20000))
    def cancel():
        raise TimeoutError("owning deadline expired")
    with pytest.raises(TimeoutError, match="owning deadline"):
        measure_manual(source, (100, 100), (19900, 19900), checkpoint=cancel)


@pytest.mark.parametrize("shape,dtype,error", [((23, 80), np.uint8, ValueError),
    ((80, 80, 4), np.uint8, ValueError), ((80, 80), np.float32, TypeError)])
def test_bad_array_rejected(kernel, shape, dtype, error):
    with pytest.raises(error):
        measure_manual(np.zeros(shape, dtype), (20, 20), (50, 50))


@pytest.mark.parametrize("point", [(-1, 50), (320, 50), (50, float("nan")), (float("inf"), 50)])
def test_bad_coordinates_rejected(kernel, strip, point):
    with pytest.raises(ValueError):
        measure_manual(strip, point, (160, 160))


def test_checkpoint_exceptions_propagate_and_kernel_remains_reusable(kernel, strip):
    class Cancelled(RuntimeError):
        pass

    for cancel_at in (1, 4, 7):
        calls = 0

        def checkpoint():
            nonlocal calls
            calls += 1
            if calls >= cancel_at:
                raise Cancelled("cancelled by the owning task")

        with pytest.raises(Cancelled, match="owning task"):
            measure_manual(strip, (160, 80), (160, 160), checkpoint=checkpoint)
        assert measure_manual(strip, (160, 80), (160, 160))["width_px"] == 19


def test_checkpoint_can_coordinate_with_other_python_threads(kernel, strip):
    requested = Event()
    confirmed = Event()

    def responder():
        assert requested.wait(2)
        confirmed.set()

    thread = Thread(target=responder)
    thread.start()

    def checkpoint():
        requested.set()
        assert confirmed.wait(2)

    result = measure_manual(strip, (160, 80), (160, 160), checkpoint=checkpoint)
    thread.join(2)
    assert result["status"] == "ok"
    assert not thread.is_alive()


def test_readonly_computation_releases_gil_without_python_checkpoint(kernel):
    source = np.full((1800, 1800), 240, np.uint8)
    source[890:910] = 30
    source.setflags(write=False)
    ready, started, finished = Event(), Event(), Event()
    def responder():
        ready.set()
        started.wait(2)
        finished.set()
    thread = Thread(target=responder)
    thread.start()
    assert ready.wait(2)
    previous = sys.getswitchinterval()
    try:
        # Prevent an ordinary Python timeslice between setting the event and
        # entering C++, so the responder needs the extension to release GIL.
        sys.setswitchinterval(1)
        started.set()
        measure_manual(source, (100, 100), (1700, 1700))
        assert finished.is_set()
    finally:
        sys.setswitchinterval(previous)
        thread.join(2)
