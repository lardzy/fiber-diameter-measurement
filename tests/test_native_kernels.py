"""Scientific equivalence and cross-language lifecycle regression tests."""
from __future__ import annotations

from dataclasses import asdict, replace
import os
from threading import Event, Thread
from time import monotonic, perf_counter, sleep

import cv2
import numpy as np
import pytest

from fdm import _native
from fdm.cancellation import CancellationError, CancellationTokenSource
from fdm.services import contour_comparison as contour
from fdm.services import fiber_quick_geometry as quick


@pytest.fixture(scope="module")
def kernels():
    try:
        return _native.require_native(validate_source=True)
    except ModuleNotFoundError as exc:
        if exc.name != _native.EXTENSION_NAME or os.environ.get("FDM_NATIVE_BACKEND") == "native":
            raise
        pytest.skip("native extension not built in this explicit source environment")


@pytest.fixture(autouse=True)
def use_native(kernels, monkeypatch):
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "native")


def masks():
    yield np.zeros((0, 0), bool)
    yield np.zeros((3, 7), bool)
    yield np.ones((1, 1), bool)
    yield np.ones((7, 11), bool)
    ring = np.zeros((35, 41), bool)
    ring[0:30, 0:27] = True
    ring[4:19, 8:17] = False
    ring[31:34, 36:40] = True
    yield ring
    yield (np.indices((29, 35)).sum(axis=0) % 2 == 0)


@pytest.mark.parametrize("mask", list(masks()))
def test_pixel_edges_exact_order_and_ownership(mask):
    baseline = contour.boundary_segments_python(mask)
    for view in (mask, mask[::-1], mask[:, ::-1], mask[::2, ::2]):
        before = view.copy()
        expected = contour.boundary_segments_python(view)
        result = contour.boundary_segments(view)
        assert result.dtype == np.float64 and result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(view, before)
        assert result.flags.owndata
    immutable = mask.copy()
    immutable.setflags(write=False)
    np.testing.assert_array_equal(contour.boundary_segments(immutable), baseline)


def test_pixel_edge_exact_limit(kernels):
    at_limit = (np.indices((1000, 1000)).sum(axis=0) % 2 == 0)
    result = contour.boundary_segments(at_limit)
    assert result.shape == (2_000_000, 2, 2)
    np.testing.assert_array_equal(result[0], [[0, 0], [1, 0]])
    del result
    too_many = np.pad(at_limit, ((0, 2), (0, 2)))
    too_many[-1, -1] = True
    for function in (contour.boundary_segments_python, contour.boundary_segments):
        with pytest.raises(ValueError, match="轮廓碎片过多"):
            function(too_many)


def assert_intervals(actual, expected):
    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected):
        assert len(row) == len(reference)
        np.testing.assert_allclose(row, reference, atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("angle", [0, .01, 17, 45, 80, 90, 135, 179])
def test_sections_rotation_holes_disconnected_and_vertices(angle):
    mask = list(masks())[4]
    axis = contour.ContourAxis((13.5, 7.5), (13.5 + np.sin(np.deg2rad(angle)), 7.5 + np.cos(np.deg2rad(angle))))
    segments = axis.to_world(contour.boundary_segments_python(mask))
    low, high = segments[:, :, 1].min(), segments[:, :, 1].max()
    for step in (.1, .5, 1., 80.):
        heights = np.arange(low, high, step, dtype=np.float64)
        expected = contour._sections_python(segments, heights, step, None)
        actual = contour._sections(segments, heights, step, None)
        assert_intervals(actual, expected)
        assert_intervals(contour._sections(segments[::-1], heights, step, None), expected)


@pytest.mark.parametrize("gap", [-1e-10, 0, 5e-9, 1e-8, 2e-8])
def test_sections_interval_merge_threshold(gap):
    # Two components: a tiny overlap is treated exactly as the reference does.
    xs = [0., 1., 1. + gap, 2.]
    segments = np.array([[[x, 0.], [x, 1.]] for x in xs])
    heights = np.array([0., .5, 1.])
    assert_intervals(contour._sections(segments, heights, .5, None), contour._sections_python(segments, heights, .5, None))


def test_sections_degenerate_and_unclosed():
    heights = np.array([0., .5])
    flat = np.array([[[0., 0.], [100., 1e-13]]])
    assert contour._sections(flat, heights, .5, None) == ((), ())
    odd = np.array([[[0., 0.], [0., 2.]]])
    for function in (contour._sections_python, contour._sections):
        with pytest.raises(ValueError, match="未闭合交点"):
            function(odd, heights, .5, None)


def skeleton_cases():
    for kind in ("empty", "line", "stairs", "cross", "short_cross", "y", "ring", "comb", "border"):
        skeleton = np.zeros((70, 80), bool)
        if kind == "line":
            skeleton[35, 8:70] = True
        elif kind == "stairs":
            for x in range(5, 60):
                skeleton[x, x] = skeleton[x, x + 1] = True
        elif kind in {"cross", "short_cross"}:
            size = 25 if kind == "cross" else 2
            skeleton[35, 40-size:41+size] = True
            skeleton[35-size:36+size, 40] = True
        elif kind == "y":
            skeleton[30:65, 40] = True
            for offset in range(21):
                skeleton[30-offset, 40-offset] = skeleton[30-offset, 40+offset] = True
        elif kind == "ring":
            skeleton[15, 15:60] = skeleton[55, 15:60] = True
            skeleton[15:56, 15] = skeleton[15:56, 59] = True
        elif kind == "comb":
            skeleton[35, 5:75] = True
            for x in range(10, 71, 10):
                skeleton[25:36, x] = True
                skeleton[24, x-2:x+3] = True
        elif kind == "border":
            skeleton[0, :] = skeleton[:, 0] = True
            skeleton[0:5, 20] = True
        yield kind, skeleton


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind,skeleton", list(skeleton_cases()))
def test_pruning_exact_masks_and_input_immutability(kind, skeleton, dtype):
    ys, xs = np.indices(skeleton.shape)
    distance = (3. + (xs + ys) / 10).astype(dtype)
    for skel, dist in ((skeleton, distance), (skeleton[::-1], distance[::-1]), (skeleton[:, ::2], distance[:, ::2])):
        before = skel.copy()
        expected = quick._prune_short_spurs_python(skel, dist)
        actual = quick._prune_short_spurs(skel, dist)
        assert actual.dtype == bool and actual.flags.owndata
        assert not np.shares_memory(actual, skel)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(skel, before)


def test_randomized_pruning_and_pixel_edges():
    rng = np.random.default_rng(20261005)
    for index in range(80):
        mask = rng.random((15 + index % 11, 18 + index % 9)) < (.1 + (index % 4) * .2)
        distance = rng.uniform(.1, 8, mask.shape).astype(np.float64 if index % 2 else np.float32)
        np.testing.assert_array_equal(quick._prune_short_spurs(mask, distance), quick._prune_short_spurs_python(mask, distance))
        np.testing.assert_array_equal(contour.boundary_segments(mask), contour.boundary_segments_python(mask))


@pytest.mark.parametrize("raw_true", [2, 128, 255])
def test_noncanonical_numpy_bool_bytes_remain_logically_true(raw_true):
    raw = np.zeros((12, 18), np.uint8)
    raw[6, 1:17] = raw_true
    raw[3:7, 9] = raw_true
    mask = raw.view(np.bool_)
    mask.setflags(write=False)
    distance = np.full(mask.shape, 2., np.float32)
    np.testing.assert_array_equal(contour.boundary_segments(mask), contour.boundary_segments_python(mask))
    np.testing.assert_array_equal(quick._prune_short_spurs(mask, distance), quick._prune_short_spurs_python(mask, distance))


def test_unaligned_distance_map_is_copied_without_precision_loss():
    skeleton = dict(skeleton_cases())["comb"]
    storage = np.zeros(skeleton.size * 8 + 1, np.uint8)
    distance = np.ndarray(skeleton.shape, dtype=np.float64, buffer=storage, offset=1)
    distance[:] = 5. + np.arange(skeleton.size).reshape(skeleton.shape) * 1e-10
    assert distance.flags.c_contiguous and not distance.flags.aligned
    expected = quick._prune_short_spurs_python(skeleton, distance)
    np.testing.assert_array_equal(quick._prune_short_spurs(skeleton, distance), expected)


def test_invalid_arrays_rejected_without_poisoning_kernel(kernels):
    mask = np.ones((4, 4), bool)
    for bad in (np.ones(5, bool), np.ones((3, 4, 2), bool)):
        with pytest.raises(ValueError):
            contour.boundary_segments(bad)
    with pytest.raises(TypeError):
        contour.boundary_segments(mask.astype(np.uint8))
    with pytest.raises(ValueError):
        quick._prune_short_spurs(mask, np.ones((4, 5), np.float32))
    with pytest.raises(TypeError):
        quick._prune_short_spurs(mask, np.ones((4, 4), np.float16))
    for bad in (np.nan, np.inf, -1.):
        with pytest.raises(ValueError):
            contour._sections(np.zeros((2, 2, 2)), np.array([0.]), bad, None)
    np.testing.assert_array_equal(contour.boundary_segments(mask), contour.boundary_segments_python(mask))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_distance_validation_includes_background_and_recovers(dtype, value):
    skeleton = np.zeros((3, 4), bool)
    distance = np.zeros(skeleton.shape, dtype=dtype)
    distance[0, 0] = value
    with pytest.raises(ValueError, match="distance_map must be finite"):
        quick._prune_short_spurs(skeleton, distance)
    # Signed zero, subnormals and the largest finite values remain accepted.
    distance[0, :4] = (-0., np.finfo(dtype).smallest_subnormal, np.finfo(dtype).max, -np.finfo(dtype).max)
    np.testing.assert_array_equal(quick._prune_short_spurs(skeleton, distance), skeleton)


def test_cancellation_exception_and_deadline_priority():
    source = CancellationTokenSource()
    source.cancel()
    with pytest.raises(CancellationError, match="操作已取消"):
        contour.boundary_segments(np.ones((4, 4), bool), token=source.token)
    mask = np.ones((5, 5), bool)
    dist = np.ones(mask.shape, np.float32)
    with pytest.raises(RuntimeError, match="请求已取消"):
        quick._prune_short_spurs(mask, dist, cancel_check=lambda: True, deadline_at=perf_counter() - 1)
    with pytest.raises(RuntimeError, match="直径计算超时"):
        quick._prune_short_spurs(mask, dist, deadline_at=perf_counter() - 1)


@pytest.mark.parametrize("operation", ["boundary", "sections", "prune"])
def test_native_callback_exception_propagates_and_recovery(kernels, operation):
    calls = []
    def checkpoint():
        calls.append(1)
        if len(calls) == 2:
            raise CancellationError("test checkpoint")
    if operation == "boundary":
        invoke = lambda cb: kernels.boundary_segments(np.ones((512, 768), bool), cb)
    elif operation == "sections":
        lines = np.repeat(np.array([[[0., 0.], [0., 200.]], [[1., 0.], [1., 200.]]]), 3000, axis=0)
        invoke = lambda cb: kernels.contour_sections(lines, np.arange(200, dtype=float), 1., cb)
    else:
        skeleton = dict(skeleton_cases())["comb"]
        invoke = lambda cb: kernels.prune_short_spurs(skeleton, np.full(skeleton.shape, 8., np.float32), cb)
    with pytest.raises(CancellationError, match="test checkpoint"):
        invoke(checkpoint)
    assert len(calls) == 2
    invoke(None)


def test_cancellation_from_another_python_thread_reaches_native(kernels):
    started = Event()
    cancelled = Event()
    times = []
    def cancel():
        if started.wait(2):
            sleep(.005)
            times.append(monotonic())
            cancelled.set()
    def checkpoint():
        started.set()
        if cancelled.is_set():
            raise CancellationError("cancelled by another thread")
    worker = Thread(target=cancel)
    mask = np.ones((6000, 6000), bool)
    worker.start()
    try:
        with pytest.raises(CancellationError, match="another thread"):
            kernels.boundary_segments(mask, checkpoint)
        assert times and monotonic() - times[0] < .1
    finally:
        worker.join(3)
    assert not worker.is_alive()


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if not key.endswith("_ms")}
    if isinstance(value, (list, tuple)):
        return [without_timing(item) for item in value]
    return value


def test_complete_measurement_results_and_baseline_rejections(monkeypatch):
    # Match the production worker's warm-up: a cold scikit-image import can
    # exceed the measurement deadline on Windows before either pruner runs.
    quick.prepare_fiber_quick_geometry_backend()
    ys, xs = np.indices((320, 320))
    service = quick.FiberQuickDiameterGeometryService()
    for width in (16, 40):
        for ratio in (3, 4, 5):
            for angle in (0, 15, 25, 45, 75):
                radians = np.deg2rad(angle)
                along = (xs-160) * np.cos(radians) + (ys-160) * np.sin(radians)
                across = -(xs-160) * np.sin(radians) + (ys-160) * np.cos(radians)
                mask = ((abs(along) < width*ratio/2) & (abs(across) < width/2)) | ((abs(across) < width*ratio/2) & (abs(along) < width/2))
                results = []
                for backend in ("python", "native"):
                    monkeypatch.setenv("FDM_NATIVE_BACKEND", backend)
                    try:
                        results.append(("ok", without_timing(asdict(service.measure_from_mask(mask)))))
                    except quick.FiberQuickGeometryError as exc:
                        results.append((exc.code, str(exc), without_timing(exc.debug_payload)))
                assert results[0] == results[1], (width, ratio, angle)


def test_complete_contour_service_uses_same_axis_and_units(monkeypatch):
    mask = list(masks())[4]
    rgba = np.full((*mask.shape, 4), 255, np.uint8)
    for angle in (0, 17, 45, 90):
        theta = np.deg2rad(angle)
        axis = contour.ContourAxis((13.5, 7.5), (13.5+np.sin(theta), 7.5+np.cos(theta)))
        frame = contour.ContourFrame("sample", rgba, mask, axis, .125)
        after = replace(frame, mask=np.roll(mask, 1, axis=0))
        monkeypatch.setenv("FDM_NATIVE_BACKEND", "python")
        expected = contour.compare_contours(frame, after, .125)
        monkeypatch.setenv("FDM_NATIVE_BACKEND", "native")
        actual = contour.compare_contours(frame, after, .125)
        assert actual.unit == expected.unit
        assert actual.warnings == expected.warnings
        assert actual.before_area == expected.before_area
        assert actual.after_area == expected.after_area
        assert len(actual.sections) == len(expected.sections)
        for row, reference in zip(actual.sections, expected.sections):
            assert row.status == reference.status and row.height == reference.height
            assert_intervals((row.before_intervals, row.after_intervals), (reference.before_intervals, reference.after_intervals))
