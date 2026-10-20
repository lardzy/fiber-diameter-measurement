"""Compare live candidate presentation with identical continuous JPEG load.

Example (run each mode in its own process):
  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/validation/benchmark_candidate_preview.py \
      --mode baseline --output /tmp/candidate-baseline.json
  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/validation/benchmark_candidate_preview.py \
      --mode guard --output /tmp/candidate-guard.json

Uses synthetic 768x576 frames and a fixed test candidate; it measures actual Qt
painting, alignment, manual acceptance and JPEG writing, NOT the automatic
measurement algorithm or camera SDK. Cocoa/Windows native runs must be reported
separately from offscreen results. It neither opens nor changes user projects.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
from fdm.live_experiment.frames import FrameHandle
from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas


def display_context(canvas):
    """Actual test surface, rather than inferring DPI from launch variables."""
    screen = canvas.screen()
    result = {"process_id": os.getpid(), "qt_scale_factor": os.environ.get("QT_SCALE_FACTOR"),
              "device_pixel_ratio": canvas.devicePixelRatioF(),
              "logical_canvas_size": [canvas.width(), canvas.height()],
              "logical_dpi": screen.logicalDotsPerInch() if screen else None,
              "physical_dpi": screen.physicalDotsPerInch() if screen else None}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
            user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
            foreground = wintypes.DWORD()
            user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(foreground))
            rect = wintypes.RECT()
            if not user32.GetClientRect(int(canvas.winId()), ctypes.byref(rect)):
                raise ctypes.WinError(ctypes.get_last_error())
            result.update(foreground_process_id=foreground.value,
                          native_client_size=[rect.right - rect.left, rect.bottom - rect.top])
        except OSError as exc:
            result["native_context_error"] = str(exc)
    return result


def windows_memory_snapshot(controller):
    """Read only this test process and its owned alignment child, once per phase."""
    if sys.platform != "win32":
        return {"available": False, "reason": "Windows working-set probe only"}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                                                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private")]
    kernel32, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    processes = {"gui": os.getpid()}
    worker = getattr(controller._alignment, "worker", None)
    child = getattr(worker, "process", None)
    if child is not None and child.is_alive():
        processes["alignment_child"] = child.pid
    result = {"available": True, "processes": {},
              "definition": "Working set/private bytes observed at phase end. Per-process peak working sets are not a simultaneous whole-application peak."}
    for role, pid in processes.items():
        handle = kernel32.OpenProcess(0x0410, False, pid)  # QUERY_INFORMATION | VM_READ
        if not handle:
            result["processes"][role] = {"pid": pid, "error": ctypes.get_last_error()}
            continue
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                result["processes"][role] = {"pid": pid, "working_set_bytes": counters.working_set,
                                            "peak_working_set_bytes": counters.peak_working_set,
                                            "private_bytes": counters.private}
            else:
                result["processes"][role] = {"pid": pid, "error": ctypes.get_last_error()}
        finally:
            kernel32.CloseHandle(handle)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "guard"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--recovery-seconds", type=float, default=0.,
                        help="After the measured load, stop new measurements but keep frames/gallery live for this many seconds.")
    parser.add_argument("--frame-ms", type=int, default=33)
    parser.add_argument("--measure-ms", type=int, default=150)
    parser.add_argument("--sample-ms", type=int, default=1,
                        help="Candidate readiness sampling interval; 0 disables this test-only timer.")
    parser.add_argument("--preview-policy", choices=("production", "immediate-probe"), default="production",
                        help="Diagnostic only: bypass preview staging while retaining the exact-frame overlay guard.")
    parser.add_argument("--gallery", action="store_true",
                        help="Keep the actual evidence gallery open and follow continuous JPEG writes.")
    args = parser.parse_args()
    if args.duration < 3 or args.frame_ms < 4 or args.measure_ms < 20 or args.sample_ms < 0 or not 0 <= args.recovery_seconds <= 60:
        parser.error("duration must be >=3s, frame-ms >=4, measure-ms >=20, sample-ms >=0 and recovery-seconds in [0,60]")
    app = QApplication([])
    data = ExperimentSession(evidence_options={"mode": "crop_jpeg"})
    data.add_operator("性能验证")
    data.set_materials(default_materials())
    scratch = tempfile.TemporaryDirectory(prefix="fdm-candidate-preview-")
    controller = LiveExperimentController(data, verify_before_present=False, evidence_directory=scratch.name)
    controller.set_calibration(CalibrationSnapshot(1., 1.))
    responses = []

    def submit(*_args, **_kwargs):
        responses.append({"task_id": "benchmark-fixture", "result": {"status": "ok",
            "candidates": [{"width_px": 20, "start": [300, 200], "end": [300, 220]}], "diagnostics": {}}})
        return "benchmark-fixture"

    def poll():
        result = list(responses)
        responses.clear()
        return result

    controller._automatic = SimpleNamespace(submit=submit, poll=poll, cancel=lambda: None, close=lambda: None)
    pixels = cv2.GaussianBlur(np.random.default_rng(293).integers(0, 256, (576, 768), np.uint8), (5, 5), 0)
    paints, paint_durations, samples, errors = [], [], [], []
    surface = None

    class Canvas(LiveMeasurementCanvas):
        def paintEvent(self, event):
            nonlocal surface
            started = time.perf_counter()
            super().paintEvent(event)
            if self.frame is not None:
                if surface is None:
                    surface = display_context(self)
                now = time.perf_counter()
                paint_durations.append((now, (now - started) * 1000))
                paints.append((now, self.frame.sequence, self._candidates_visible(), now - self.frame.received_at))

    canvas = Canvas(controller)
    controller.frameReady.connect(canvas.set_frame)
    canvas.resize(768, 576)
    canvas.show()
    gallery = None
    gallery_peak_bytes = 0
    gallery_visible_refreshes = gallery_hidden_refreshes = 0
    if args.gallery:
        from fdm.ui.measurement_evidence_gallery import EvidenceGalleryController
        gallery = EvidenceGalleryController(canvas, preferences_path=Path(scratch.name) / "gallery.json")
        gallery.bind(SimpleNamespace(session=data, evidence_review=SimpleNamespace(show_record=lambda *_a, **_k: None)))
        gallery.open()

    def refresh_gallery():
        nonlocal gallery_peak_bytes, gallery_visible_refreshes, gallery_hidden_refreshes
        if gallery.dialog.isVisible():
            gallery_visible_refreshes += 1
        else:
            gallery_hidden_refreshes += 1
        gallery.refresh()
        gallery_peak_bytes = max(gallery_peak_bytes, gallery.memory_bytes)

    sequence, attempts = 0, 0
    load_finished_at = None
    recovery_start = None
    load_memory = None

    def receive():
        nonlocal sequence
        sequence += 1
        value = FrameHandle.snapshot(pixels, source_id="benchmark-synthetic", sequence=sequence, live=True)
        if controller.receive_frame(value):
            if args.preview_policy == "immediate-probe":
                controller._publish_preview(value)
            else:
                controller.present_preview_frame(value)

    def measure():
        nonlocal attempts
        if load_finished_at is not None or controller.state != "ready" or controller.presented is None:
            return
        try:
            x, y = 40 + (attempts % 20) * 24, 350 + ((attempts // 20) % 6) * 25
            controller.start((x, y))
            controller.finish((x, y + 20), "wool", method="free")
            attempts += 1
        except ValueError as exc:
            errors.append(str(exc))

    timers = []
    timer_specs = [(args.frame_ms, receive), (args.measure_ms, measure)]
    if gallery is not None:
        controller.changed.connect(refresh_gallery)
        timer_specs.append((100, refresh_gallery))
    if args.sample_ms:
        timer_specs.append((args.sample_ms, lambda: samples.append((time.perf_counter(), controller.candidate_overlay_ready))))
    for interval, callback in timer_specs:
        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(interval)
        timer.timeout.connect(callback)
        timer.start()
        timers.append(timer)
    if args.mode == "guard":
        QTimer.singleShot(350, lambda: controller.analyze())

    def finish_load():
        nonlocal load_finished_at, recovery_start, load_memory
        load_finished_at = time.perf_counter()
        # Only stop the producer of new business work; outstanding acceptance,
        # evidence writes, source frames and gallery decoding still finish.
        timers[1].stop()
        recovery_start = {"at": load_finished_at, "accepted": len(data.measurements),
                          "saved": sum(r.evidence.get("state") == "saved" for r in data.measurements.values()),
                          "measurement_attempts": attempts}
        load_memory = windows_memory_snapshot(controller)
        if args.recovery_seconds:
            QTimer.singleShot(round(args.recovery_seconds * 1000), app.quit)
        else:
            app.quit()

    QTimer.singleShot(round(args.duration * 1000), finish_load)
    try:
        app.exec()
    finally:
        for timer in timers:
            timer.stop()
        gallery_result = None
        if gallery is not None:
            gallery_result = {"visible_at_event_loop_exit": gallery.dialog.isVisible(),
                              "visible_refreshes": gallery_visible_refreshes,
                              "hidden_refreshes": gallery_hidden_refreshes,
                              "visibility_note": "QApplication.quit closes windows before app.exec returns; running visibility is counted by refreshes.",
                              "decoded": gallery.decode_count,
                              "records": gallery.proxy.rowCount(), "peak_memory_bytes": gallery_peak_bytes,
                              "limit_bytes": gallery.MEMORY_LIMIT}
            gallery.close()
            gallery_result["remaining_memory_bytes"] = gallery.memory_bytes
            controller.changed.disconnect(refresh_gallery)
        controller.close()
        canvas.close()
    if not paints:
        raise RuntimeError("No frames were painted; benchmark is invalid")
    start = paints[0][0] + 1.5
    recovery_paints = [item for item in paints if load_finished_at is not None and item[0] >= load_finished_at]
    # Keep the existing headline FPS/age comparison scoped to the unchanged
    # business-load interval; a quiet recovery tail must not improve its P95.
    stop = load_finished_at if load_finished_at is not None else float("inf")
    samples = [item for item in samples if start < item[0] < stop]
    paints = [item for item in paints if start < item[0] < stop]
    paint_durations = [duration for when, duration in paint_durations if start < when < stop]
    first = {}
    for when, seq, ready, age in paints:
        first.setdefault(seq, (when, age))
    frames = list(first.values())
    if len(frames) < 2:
        raise RuntimeError("Too few frames after warmup; benchmark is invalid")
    result = {
        "scope": "synthetic frames; fixed analysis fixture; actual Qt/alignment/acceptance/JPEG; not camera or algorithm acceptance",
        "mode": args.mode, "qt_platform": app.platformName(), "frame_size": [768, 576],
        "frame_interval_ms": args.frame_ms, "measure_interval_ms": args.measure_ms,
        "sample_interval_ms": args.sample_ms,
        "preview_policy": args.preview_policy,
        "display_context": surface, "windows_memory_at_load_end": load_memory,
        "opencv_threads": cv2.getNumThreads(), "warmup_seconds": 1.5,
        "frames": len(frames), "fps": (len(frames) - 1) / (frames[-1][0] - frames[0][0]),
        "paint_count": len(paints), "hidden_paints": sum(not item[2] for item in paints),
        "paint_duration_ms": {"n": len(paint_durations), "p50": float(np.percentile(paint_durations, 50)),
                              "p95": float(np.percentile(paint_durations, 95))},
        "ready_sample_fraction": sum(item[1] for item in samples) / len(samples) if samples else None,
        "p95_frame_age_ms": float(np.percentile([item[1] * 1000 for item in frames], 95)),
        "accepted": len(data.measurements),
        "saved": sum(record.evidence.get("state") == "saved" for record in data.measurements.values()),
        "jpeg_bytes": sum(path.stat().st_size for path in Path(scratch.name).glob("*.jpg")),
        "peak_budget_bytes": controller.budget.peak_bytes,
        "remaining_budget_bytes": controller.budget.used_bytes,
        "metrics": controller.performance_diagnostics(),
        "metrics_schema": controller.performance_diagnostics_schema(),
        "frame_sampling": controller.performance_frame_context(), "measurement_errors": errors,
        "gallery": gallery_result,
    }
    if args.recovery_seconds:
        visible = [p for p in recovery_paints if p[2]]
        trailing = [p for p in recovery_paints if p[0] >= stop + max(0., args.recovery_seconds - 1.)]
        changes = [b for a, b in zip(recovery_paints, recovery_paints[1:]) if a[2] != b[2]]
        result["post_load_recovery"] = {
            "requested_seconds": args.recovery_seconds,
            "started_at_perf_counter": stop,
            "observed_seconds": recovery_paints[-1][0] - stop if recovery_paints else 0.,
            "start": recovery_start,
            "accepted_at_end": len(data.measurements), "saved_at_end": result["saved"],
            "measurement_attempts_at_end": attempts,
            "paint_count": len(recovery_paints), "hidden_paints": sum(not p[2] for p in recovery_paints),
            "first_visible_after_ms": (visible[0][0] - stop) * 1000 if visible else None,
            "visibility_transitions": [{"seconds": p[0] - stop, "sequence": p[1], "visible": p[2]} for p in changes],
            "last_second_continuously_visible": bool(trailing) and all(p[2] for p in trailing),
            "definition": "New measurement timer stopped; source frames, pending acceptance, JPEG writes and gallery remain active. Load FPS/P95 above exclude this tail; original continuous-visibility acceptance is not relaxed.",
        }
    if gallery_result is not None:
        gallery_result["passed"] = bool(
            gallery_result["visible_refreshes"] > 0 and gallery_result["hidden_refreshes"] == 0
            and gallery_result["decoded"] > 0 and gallery_result["records"] == result["accepted"]
            and gallery_result["peak_memory_bytes"] <= gallery_result["limit_bytes"]
            and gallery_result["remaining_memory_bytes"] == 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    scratch.cleanup()
    return int(bool(errors or result["accepted"] != result["saved"] or result["remaining_budget_bytes"]
                    or gallery_result is not None and not gallery_result["passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
