"""Synthetic 30 Hz input through the real MainWindow and capture manager.

Run serially, after correctness tests. Measures Qt delivery/painting and actual
two-click coordinates while switching live/content/diameter pages. This is not
camera, native SDK or display scan-out acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scripts.benchmark_cu_live_measurement import distribution, environment, fixture, input_digest
from scripts.validation import validate_workflow_redesign as workflow
from scripts.validation.gui_resource_sampler import ProcessResourceSampler
import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from fdm.live_experiment.controller import CalibrationSnapshot
from fdm.live_experiment.frames import FrameHandle
import fdm.ui.live_experiment_workspace as trial_module


class StreamingBackend(workflow.FakeCaptureBackend):
    def __init__(self, pixels, hz):
        super().__init__(size=(pixels.shape[1], pixels.shape[0]))
        self.pixels, self.hz = pixels, hz
        self.stop_event = threading.Event()
        self.producer = None
        self.generated = 0
        self.errors = []

    def start_preview(self, device, **kwargs):
        super().start_preview(device, **kwargs)
        callback = kwargs["frame_callback"]
        self.stop_event.clear()

        def produce():
            due = time.perf_counter()
            try:
                while not self.stop_event.is_set():
                    pixels = np.array(self.pixels, copy=True, order="C")
                    # Distinct low-amplitude markers prevent an accidental
                    # frozen-frame cache hit without moving the measured fiber.
                    for bit in range(16):
                        pixels[12:18, 12 + bit * 6:18 + bit * 6] = 240 + ((self.generated >> bit) & 1)
                    frame = FrameHandle.snapshot(pixels, transfer_ownership=True,
                        source_id="workflow-performance", sequence=self.generated,
                        metadata={"synthetic": True, "benchmark_sequence": self.generated})
                    self.generated += 1
                    callback(frame)
                    due = max(due + 1 / self.hz, time.perf_counter())
                    self.stop_event.wait(max(0, due - time.perf_counter()))
            except Exception as exc:
                self.errors.append(repr(exc))

        self.producer = threading.Thread(target=produce, name="workflow-performance-source", daemon=True)
        self.producer.start()

    def stop_preview(self):
        self.stop_event.set()
        if self.producer is not None:
            self.producer.join(5)
            assert not self.producer.is_alive(), "synthetic producer failed to stop"
            self.producer = None
        super().stop_preview()


def wait_for(condition, timeout=5):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        QApplication.processEvents()
        if condition():
            return
        time.sleep(.001)
    raise AssertionError("Timed out waiting for an actual frame or measurement")


def pump(seconds):
    deadline = time.perf_counter() + seconds
    wait_for(lambda: time.perf_counter() >= deadline, seconds + 1)


def run_case(args, width):
    pixels, start, end = fixture(width)
    backend = StreamingBackend(pixels, args.frame_hz)
    phase = [None]
    original_canvas = trial_module.LiveMeasurementCanvas

    class MeasuredCanvas(original_canvas):
        def paintEvent(self, event):
            began = time.perf_counter()
            super().paintEvent(event)
            now = time.perf_counter()
            if phase[0] is not None and self.isVisible() and self.frame is not None:
                phase[0]["paint_ms"].append((now - began) * 1000)
                phase[0]["painted"].setdefault(self.frame.frame_id, (now, self.frame.received_at))

    with tempfile.TemporaryDirectory(prefix="fdm-workflow-perf-") as profile, \
            patch.object(trial_module, "LiveMeasurementCanvas", MeasuredCanvas), \
            workflow.WorkflowHarness(profile, theme=args.theme, backend=backend) as harness:
        window = harness.window
        harness.enter(frame=False)
        window._preview_display_binding.set_mode("fit")
        wait_for(lambda: window._capture_manager.latest_preview_frame() is not None)
        heartbeat = []
        last_beat = [time.perf_counter()]
        timer = QTimer(window)
        timer.setInterval(10)
        def beat():
            now = time.perf_counter()
            if phase[0] is not None:
                heartbeat.append((now - last_beat[0]) * 1000)
            last_beat[0] = now
        timer.timeout.connect(beat)
        timer.start()
        def delivered(_image):
            frame = window._capture_manager.latest_preview_frame()
            if phase[0] is not None and frame is not None:
                phase[0]["delivered"].add(frame.sequence)
        window._capture_manager.frameReady.connect(delivered)
        phases, coordinates = [], []
        with ProcessResourceSampler(parent_only=True) as resources:
            for index in (0, 1, 2, 0, 1, 2):
                current = {"page": ("realtime", "content", "diameter")[index],
                    "started": time.perf_counter(), "generated": backend.generated,
                    "paint_ms": [], "painted": {}, "delivered": set()}
                phase[0] = current
                selected = harness.select(index, frame=False)
                current["navigation_ms"] = (time.perf_counter() - current["started"]) * 1000
                if index:
                    selected.session.save_images = False
                    selected.controller.set_calibration(CalibrationSnapshot(1, 1, 1, "synthetic coordinate probe"))
                    selected.display_binding.set_mode("fit")
                    controller = selected.controller
                    rejected = []
                    on_rejected = rejected.append
                    controller.resultRejected.connect(on_rejected)
                    wait_for(lambda: controller.state == "ready" and controller.presented is not None)
                    token = controller.presented
                    first = QPoint(round(start[0] * token.scale + token.offset[0]),
                                   round(start[1] * token.scale + token.offset[1]))
                    QTest.mouseClick(selected.canvas, Qt.MouseButton.LeftButton, pos=first)
                    assert controller.draft is not None
                    draft_token = controller.draft[0]
                    painted_before = len(current["painted"])
                    pump(.2)
                    frames_during_draft = len(current["painted"]) - painted_before
                    assert frames_during_draft > 0, "the draft blocked live painting"
                    second = QPoint(round(end[0] * draft_token.scale + draft_token.offset[0]),
                                    round(end[1] * draft_token.scale + draft_token.offset[1]))
                    previous_ids = set(selected.session.measurements)
                    click_at = time.perf_counter()
                    QTest.mouseClick(selected.canvas, Qt.MouseButton.LeftButton, pos=second)
                    wait_for(lambda: bool(set(selected.session.measurements) - previous_ids) or bool(rejected))
                    assert not rejected, "Measurement rejected: " + "; ".join(rejected)
                    controller.resultRejected.disconnect(on_rejected)
                    record = selected.session.measurements[next(iter(set(selected.session.measurements) - previous_ids))]
                    errors = [abs(actual - expected) for key, expected_point in (("input_start", start), ("input_end", end))
                              for actual, expected in zip(record.provenance[key], expected_point)]
                    tolerance = .5 / draft_token.scale + 1e-6
                    assert max(errors) <= tolerance
                    assert record.frame_id == draft_token.frame.frame_id
                    assert abs(record.diameter_um - 19) < .05
                    coordinates.append({"page": current["page"], "frame_id_matches_draft": True,
                        "diameter_um": record.diameter_um, "expected_diameter_um": 19,
                        "max_input_coordinate_error_px": max(errors), "rounding_tolerance_px": tolerance,
                        "frames_painted_during_200ms_draft": frames_during_draft,
                        "second_click_to_accept_ms": (time.perf_counter() - click_at) * 1000})
                pump(max(0, args.phase_seconds - (time.perf_counter() - current["started"])))
                elapsed = time.perf_counter() - current["started"]
                paints = list(current["painted"].values())
                phases.append({"page": current["page"], "elapsed_seconds": elapsed,
                    "navigation_ms": current["navigation_ms"],
                    "generated_fps": (backend.generated - current["generated"]) / elapsed,
                    "manager_delivered_fps": len(current["delivered"]) / elapsed,
                    "distinct_trial_painted_fps": len(paints) / elapsed if index else None,
                    "first_trial_paint_ms": (paints[0][0] - current["started"]) * 1000 if paints else None,
                    "trial_frame_age_ms": distribution([(done - received) * 1000 for done, received in paints]),
                    "trial_paint_duration_ms": distribution(current["paint_ms"])})
                phase[0] = None
                harness.assert_size()
                harness.assert_navigation()
        timer.stop()
        assert not backend.errors and not harness.slot_errors and not harness.modal_messages
        assert backend.starts == 1, "switching pages restarted the device"
        return {"frame_size": [width, pixels.shape[0]], "input_sha256": input_digest(pixels),
            "window_size": [window.width(), window.height()], "dpr": window.devicePixelRatioF(),
            "phases": phases, "coordinate_checks": coordinates, "backend_starts": backend.starts,
            "gui_heartbeat_interval_ms": distribution(heartbeat), "resources": resources.result,
            "producer_errors": backend.errors, "slot_errors": harness.slot_errors,
            "boundary": "synthetic producer -> actual capture manager -> real MainWindow; trial paint timings only; realtime fps counts manager delivery, not scan-out"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--frame-hz", type=float, default=30)
    parser.add_argument("--phase-seconds", type=float, default=2)
    parser.add_argument("--theme", choices=("dark", "light", "navy", "warm"), default="dark")
    args = parser.parse_args()
    if min(args.sizes) < 320 or min(args.frame_hz, args.phase_seconds) <= 0:
        parser.error("positive rates and dimensions >=320 are required")
    report = {"environment": environment(), "parameters": vars(args).copy(),
              "cases": [], "correctness_checks_passed": False}
    for relative in ("src/fdm/ui/main_window.py", "src/fdm/ui/live_experiment_workspace.py",
                     "src/fdm/ui/live_experiment_view.py", "src/fdm/ui/acquisition_workbench.py",
                     "src/fdm/ui/workspace_stack.py", "src/fdm/ui/theme.py", "src/fdm/ui/icons.py",
                     "scripts/validation/validate_workflow_redesign.py", "scripts/validation/benchmark_workflow_redesign.py",
                     "scripts/validation/gui_resource_sampler.py"):
        report["environment"]["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for width in args.sizes:
        print(f"MainWindow {width}px @ {args.frame_hz} Hz", flush=True)
        try:
            report["cases"].append(run_case(args, width))
        except Exception as exc:
            report["failed_case"] = {"frame_width": width, "error": str(exc),
                "traceback": traceback.format_exc(), "status": "incomplete"}
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            raise
        report["correctness_checks_passed"] = len(report["cases"]) == len(args.sizes)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
