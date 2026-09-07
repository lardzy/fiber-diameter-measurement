import multiprocessing
import sys
import time

import pytest

from fdm.ui import overlay_process_renderer as renderer
from fdm.ui import overlay_render_self_check as probe


def _unfixed_windowed_initializer():
    import faulthandler

    sys.stderr = sys.__stderr__ = None
    faulthandler.enable()


def _blank_raster(payload):
    from fdm.ui.overlay_process_renderer import _render

    image, picture = _render(payload)
    if image is not None:
        image = (*image[:-1], bytes(len(image[-1])))
    return image, picture


def _stuck_raster(_payload):
    time.sleep(60)


def test_real_windowed_probe_checks_all_pixels_without_touching_live_pool(monkeypatch):
    live_pool = object()
    monkeypatch.setattr(renderer, "_pool", live_pool)
    report = probe.run_overlay_render_self_check()

    assert renderer._pool is live_pool
    assert report["ok"] and report["worker_stdio_none"]
    assert report["worker_platform"] == "offscreen"
    assert report["start_method"] == "spawn"
    assert set(report["cases"]) == {
        f"{name}@{dpr}"
        for name in (
            "magic_primary",
            "magic_subtract",
            "scene_overview",
            "mixed_exact",
            "empty",
        )
        for dpr in ("1", "1.5", "2")
    }
    assert all(case["ok"] for case in report["cases"].values())
    assert all(
        len(case["sha256"]) == 64
        for case in report["cases"].values()
        if not case.get("empty")
    )
    assert report["worker_pid"] not in {
        process.pid for process in multiprocessing.active_children()
    }


def test_probe_catches_the_original_windowed_initializer_failure(monkeypatch):
    monkeypatch.setattr(
        probe, "_initialize_windowed_worker", _unfixed_windowed_initializer
    )
    with pytest.raises(RuntimeError, match="worker startup:.*terminated abruptly"):
        probe.run_overlay_render_self_check()


def test_probe_rejects_a_worker_that_returns_blank_images(monkeypatch):
    monkeypatch.setattr(renderer, "_render", _blank_raster)
    with pytest.raises(RuntimeError, match="magic_primary@1: raster pixels differ"):
        probe.run_overlay_render_self_check()


def test_probe_times_out_and_reaps_a_stuck_worker(monkeypatch):
    monkeypatch.setattr(renderer, "_render", _stuck_raster)
    previous = {process.pid for process in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(
        RuntimeError, match="magic_primary@1: overlay renderer timed out"
    ):
        probe.run_overlay_render_self_check(timeout_seconds=3)
    assert time.monotonic() - started < 8
    assert {process.pid for process in multiprocessing.active_children()} <= previous
