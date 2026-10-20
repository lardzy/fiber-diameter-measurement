"""Record real Qt candidate painting before/after a scheduling-pressure fix.

The candidate measurements and immutable source pixels are deterministic test
fixtures. Rendering, preview delivery, alignment, timing, and all visibility
decisions run through the selected production source. A 60 Hz QWidget capture
adds overhead: use validate_candidate_presentation.py for performance numbers.
This is an offscreen reproduction, not a recording of a Microview device.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--delay-ms", type=int, default=80)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve() / "src"))

    import cv2
    import numpy as np
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    import fdm.live_experiment.candidate_alignment as alignment
    from fdm.live_experiment import ExperimentSession, default_materials
    from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
    from fdm.live_experiment.frames import FrameHandle
    from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas

    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    session = ExperimentSession(evidence_options={"mode": "disabled"})
    session.add_operator("Recording fixture")
    session.set_materials(default_materials())
    controller = LiveExperimentController(session, verify_before_present=False)
    controller.set_calibration(CalibrationSnapshot(1., 1.))
    candidates = []
    noise = np.random.default_rng(293).integers(0, 256, (576, 768), np.uint8)
    pixels = cv2.GaussianBlur(noise, (5, 5), 0)
    for index, x in enumerate((85, 205, 325, 445, 565, 685)):
        y = 125 + (index % 3) * 125
        cv2.line(pixels, (x - 25, 20), (x + 25, 555), 35, 22)
        center = x - 25 + (y - 20) * 50 / 535
        candidates.append({"width_px": 22, "start": [center - 11, y], "end": [center + 11, y]})
    pending = []

    def submit(*_args, **_kwargs):
        pending.append({"task_id": "recording-fixture", "result": {
            "status": "ok", "candidates": candidates, "diagnostics": {}}})
        return "recording-fixture"

    def poll():
        result = list(pending)
        pending.clear()
        return result

    controller._automatic = SimpleNamespace(submit=submit, poll=poll, cancel=lambda: None, close=lambda: None)
    failures = []
    controller.resultRejected.connect(failures.append)
    # Inject at the monitor boundary so the delay also covers spawned workers.
    work = alignment.CandidateAlignmentMonitor._work
    delay_injections = []
    start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    def delayed(monitor, *values, **kwargs):
        result = work(monitor, *values, **kwargs)
        if args.delay_ms and 2. <= time.perf_counter() - start < 3.5:
            wait_start = time.perf_counter()
            time.sleep(args.delay_ms / 1000.)
            waited_ms = (time.perf_counter() - wait_start) * 1000
            delay_injections.append({"started_seconds": wait_start - start, "elapsed_ms": waited_ms})
            return replace(result, elapsed_ms=result.elapsed_ms + waited_ms)
        return result

    alignment.CandidateAlignmentMonitor._work = delayed
    paints, captures, images, batch_ids = [], [], [], set()

    class Canvas(LiveMeasurementCanvas):
        def paintEvent(self, event):
            super().paintEvent(event)
            if self.frame is not None:
                paints.append({"seconds": time.perf_counter() - start,
                               "sequence": self.frame.sequence,
                               "visible": bool(self._candidates_visible())})
            context = controller._automatic_context
            if context and context.get("batch_id"):
                batch_ids.add(context["batch_id"])

    window = QWidget()
    window.setWindowTitle("Actual Qt candidate presentation reproduction")
    layout = QVBoxLayout(window)
    title = QLabel(f"{args.label.upper()} | real Qt rendering | fixed source at 30 Hz")
    title.setFixedHeight(26)
    layout.addWidget(title)
    canvas = Canvas(controller)
    canvas.setFixedSize(768, 576)
    layout.addWidget(canvas)
    state = QLabel()
    state.setFixedHeight(26)
    layout.addWidget(state)
    controller.frameReady.connect(canvas.set_frame)
    window.show()
    sequence = 0

    def receive():
        nonlocal sequence
        sequence += 1
        frame = FrameHandle.snapshot(pixels, source_id="recording-synthetic", sequence=sequence, live=True)
        if controller.receive_frame(frame):
            controller.present_preview_frame(frame)

    def capture():
        seconds = time.perf_counter() - start
        visible = bool(canvas._candidates_visible())
        pressure = 2. <= seconds < 3.5
        state.setText(f"t={seconds:.2f}s | candidates {'VISIBLE' if visible else 'HIDDEN'} | "
                      f"alignment delay: {args.delay_ms if pressure else 0} ms")
        image = window.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        image_array = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        rgb = image_array[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
        # Encode only after the event loop has ended: avoid disk/codec pressure
        # changing which production frame is eligible to display.
        images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        captures.append({"seconds": seconds, "visible": bool(canvas._candidates_visible()),
                         "sequence": canvas.frame.sequence if canvas.frame else None})

    timers = []
    for interval, callback in ((33, receive), (16, capture)):
        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(interval)
        timer.timeout.connect(callback)
        timer.start()
        timers.append(timer)
    def analyze():
        try:
            controller.analyze()
        except (ValueError, RuntimeError) as exc:
            failures.append(str(exc))
    QTimer.singleShot(350, analyze)
    QTimer.singleShot(round(args.duration * 1000), app.quit)
    try:
        app.exec()
    finally:
        for timer in timers:
            timer.stop()
        controller.close()
        window.close()
        alignment.CandidateAlignmentMonitor._work = work

    if failures or len(batch_ids) != 1 or (args.delay_ms and not delay_injections):
        raise RuntimeError(f"Recording fixture did not produce one stable batch: {failures}, batches={len(batch_ids)}")

    # Keep recording playback close to actual elapsed time; each emitted frame
    # is an actual Qt capture. No candidate pixels are redrawn for the video.
    fps = (len(captures) - 1) / (captures[-1]["seconds"] - captures[0]["seconds"])
    height, width = images[0].shape[:2]
    output = args.output / f"{args.label}.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Video encoder did not open")
    for frame in images:
        writer.write(frame)
    writer.release()
    transitions = [b for a, b in zip(paints, paints[1:]) if a["visible"] != b["visible"]]
    report = {
        "scope": "Actual Qt widget capture of synthetic fixed source and fixed candidate fixture; no camera or native menu test.",
        "capture_overhead": "Widget capture at about 60 Hz adds paints and memory copies. Not a performance benchmark.",
        "label": args.label, "started_at": started_at, "source_root": str(args.source_root.resolve()),
        "platform": app.platformName(), "opencv_threads": cv2.getNumThreads(),
        "injection": {"delay_ms": args.delay_ms, "seconds": [2., 3.5], "actual": delay_injections},
        "fps": fps, "captures": captures, "paints": paints, "batch_count": len(batch_ids),
        "visibility_transitions": transitions, "transition_count": len(transitions),
        "remaining_budget_bytes": controller.budget.used_bytes,
        "source_sha256": {name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() for name in (
            "src/fdm/live_experiment/controller.py", "src/fdm/live_experiment/candidate_alignment.py",
            "src/fdm/live_experiment/performance.py", "src/fdm/ui/live_measurement_canvas.py")},
        "loaded_fdm_sources": {str(Path(module.__file__).resolve().relative_to(args.source_root.resolve())):
                               hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                               for name, module in tuple(sys.modules.items())
                               if name.startswith("fdm.") and getattr(module, "__file__", None)
                               and Path(module.__file__).suffix == ".py"
                               and Path(module.__file__).resolve().is_relative_to(args.source_root.resolve())},
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (args.output / f"{args.label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for target in (1., 2.5, 4.5, 6.5):
        index = min(range(len(captures)), key=lambda i: abs(captures[i]["seconds"] - target))
        cv2.imwrite(str(args.output / f"{args.label}-{target:.1f}s.png"), images[index])
    print(json.dumps({"video": str(output), "transition_count": len(transitions), "fps": fps,
                      "batch_count": len(batch_ids), "remaining_budget_bytes": controller.budget.used_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
