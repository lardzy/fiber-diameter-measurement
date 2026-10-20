"""Offline timings retain exact definitions, bounded scalars and no frame data."""
import json
from types import SimpleNamespace

import pytest

from fdm.live_experiment.performance import LivePerformanceDiagnostics, METRICS


def frame(sequence, received_at, *, generation=1, live=True):
    return SimpleNamespace(source_id="camera", generation=generation, size=(768, 576),
                           sequence=sequence, received_at=received_at, live=live)


def test_rates_count_distinct_frames_and_age_ends_at_first_paint():
    metrics = LivePerformanceDiagnostics()
    first, second = frame(1, 10.), frame(3, 10.1)
    metrics.received(first, 10.01)
    metrics.presented(first, 10.02)
    metrics.presented(first, 10.09)  # Repainting labels must not inflate FPS/age.
    metrics.received(second, 10.11)
    metrics.presented(second, 10.13)
    result = metrics.summary()
    assert result["ingest_fps"]["n"] == 1
    assert result["ingest_fps"]["p50"] == pytest.approx(10)
    assert result["presented_fps"]["p50"] == pytest.approx(1 / .11)
    assert result["frame_age_ms"] == pytest.approx({"n": 2, "p50": 25., "p95": 29.5})
    context = metrics.frame_context()
    assert context["presented"]["frames"] == 2
    assert context["presented"]["window_fps"] == pytest.approx(1 / .11)


def test_stop_retains_last_run_and_reopen_does_not_count_the_stopped_gap():
    metrics = LivePerformanceDiagnostics()
    for i in range(3):
        value = frame(i, 10 + i)
        metrics.received(value, 10 + i)
        metrics.presented(value, 10.1 + i)
    metrics.record("analysis_ms", 25)
    before = metrics.summary()
    metrics.stop()
    assert metrics.summary() == before
    assert metrics.frame_context()["stopped"]
    value = frame(10, 100)
    metrics.received(value, 100)
    metrics.presented(value, 100.01)
    assert metrics.summary()["ingest_fps"]["n"] == 0
    assert metrics.summary()["frame_age_ms"]["n"] == 1
    assert metrics.summary()["analysis_ms"]["n"] == 1
    metrics.received(frame(0, 101, generation=2), 101)
    metrics.presented(value, 102)  # Late old-generation paint is not sampled.
    assert metrics.summary()["frame_age_ms"]["n"] == 0


def test_static_images_do_not_produce_capture_rates_and_future_ages_are_rejected():
    metrics = LivePerformanceDiagnostics()
    static = frame(1, 1, live=False)
    metrics.received(static, 1)
    metrics.presented(static, 1.1)
    assert metrics.summary()["frame_age_ms"]["n"] == 0
    value = frame(1, 20, generation=2)
    metrics.received(value, 2)
    metrics.presented(value, 2.1)
    assert metrics.summary()["frame_age_ms"]["n"] == 0
    assert metrics.frame_context()["invalid_frame_age_samples"] == 1


def test_samples_are_bounded_and_diagnostics_remain_json_scalars():
    metrics = LivePerformanceDiagnostics()
    for i in range(600):
        value = frame(i, i / 30)
        metrics.received(value, i / 30)
        metrics.presented(value, i / 30 + .01)
        metrics.record("analysis_ms", i)
    assert all(len(samples) <= 512 for samples in metrics.samples.values())
    assert all(len(stream) == 512 for stream in metrics.streams.values())
    assert metrics.summary()["frame_age_ms"]["n"] == 512
    assert metrics.frame_context()["presented"]["first_sequence"] == 88
    assert set(metrics.schema()["metrics"]) == set(METRICS)
    json.dumps({"schema": metrics.schema(), "metrics": metrics.summary(),
                "frames": metrics.frame_context()}, allow_nan=False)


def test_analysis_exports_service_and_combined_overhead_without_inventing_absent_stages():
    metrics = LivePerformanceDiagnostics()
    metrics.analysis({"result": {"diagnostics": {"total_ms": 21, "preparation_ms": 1}},
                      "timings": {"shared_memory_ms": 2, "budget_wait_ms": 3,
                                  "dispatch_to_receive_ms": 25, "startup_transport_overhead_ms": 4}})
    result = metrics.summary()
    assert result["analysis_service_total_ms"]["p50"] == 21
    assert result["analysis_preparation_ms"]["p50"] == 1
    assert result["analysis_processing_ms"]["p50"] == 20
    assert result["analysis_startup_transport_overhead_ms"]["p50"] == 4
    assert result["analysis_process_start_call_ms"] == {"n": 0, "p50": None, "p95": None}
    metrics.analysis({"result": {"diagnostics": {"total_ms": float("nan")}},
                      "timings": {"shared_memory_ms": -1, "budget_wait_ms": float("inf")}})
    assert metrics.summary() == result
