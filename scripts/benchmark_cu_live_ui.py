"""Real Qt paint/event-loop CU measurements with a synthetic 30 Hz live source.

    python scripts/benchmark_cu_live_ui.py --output results/cu-live-ui.json

Runs 2K/4K by default, five warmups and thirty QTest two-click measurements, with PNG
saving off and on. Frames enter the actual latest-only workspace mailbox from
a producer thread; real QWidget paintEvent and controller kernels run. This is
an offscreen rendering benchmark, not screen-presentation or camera acceptance.
Run serially after correctness tests; do not overlap with other benchmarks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from scripts.benchmark_cu_live_measurement import ResourceSampler, distribution, environment, fixture, input_digest
import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from fdm.live_experiment import ExperimentSession, default_materials
from fdm.live_experiment.controller import CalibrationSnapshot
from fdm.live_experiment.frames import FrameHandle
from fdm.models import ProjectState
import fdm.ui.live_experiment_workspace as workspace_module


class BenchmarkHost(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.project = ProjectState.empty()
        self.project.live_experiments.append(session)
        self._images = {}
        self.save_project_action = QAction(self)

    def current_document(self):
        return None


def wait_until(condition, seconds, *, check=None):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        QApplication.processEvents()
        if check:
            check()
        if condition():
            return True
        time.sleep(.001)
    return False


def run_case(args, size, save_images):
    session = ExperimentSession(save_images=save_images, sample_name=f"UI synthetic {size}")
    session.add_operator("benchmark")
    session.set_materials(default_materials())
    host = BenchmarkHost(session)
    state = {"paints": 0, "paint_ms": [], "current_trial": None, "painted_frame_ids": set(),
             "sample_start": None, "sample_end": None, "sample_frame_ids": set(), "frame_age_ms": []}
    original_canvas = workspace_module.LiveMeasurementCanvas

    class MeasuredCanvas(original_canvas):
        def paintEvent(self, event):
            started = time.perf_counter()
            super().paintEvent(event)
            done = time.perf_counter()
            state["paints"] += 1
            state["paint_ms"].append((done - started) * 1000)
            if self.frame is not None:
                state["painted_frame_ids"].add(self.frame.frame_id)
                if state["sample_start"] is not None and state["sample_end"] is None:
                    if self.frame.frame_id not in state["sample_frame_ids"]:
                        state["sample_frame_ids"].add(self.frame.frame_id)
                        state["frame_age_ms"].append((done - self.frame.received_at) * 1000)
            trial = state["current_trial"]
            if trial is None:
                return
            if self.controller.draft and "draft_painted_at" not in trial:
                trial["draft_painted_at"] = done
            if trial.get("record_id") in self.controller.current_records and "record_painted_at" not in trial:
                trial["record_painted_at"] = done

    workspace_module.LiveMeasurementCanvas = MeasuredCanvas
    try:
        workspace = workspace_module.open_live_experiment(host)
    finally:
        workspace_module.LiveMeasurementCanvas = original_canvas
    controller = workspace.controller
    # Use the actual EvidenceStore; redirect its already-created empty directory.
    evidence_directory = tempfile.TemporaryDirectory(prefix="fdm-cu-ui-evidence-")
    original_evidence_directory = controller.evidence.directory
    controller.evidence.directory = Path(evidence_directory.name)
    original_evidence_directory.rmdir()
    controller.set_calibration(CalibrationSnapshot(1, 1, 1, "synthetic UI benchmark"))
    workspace.resize(args.window_width, args.window_height)
    workspace.activateWindow()
    pixels, start, end = fixture(size)
    # Alternate two precomputed sensor states so the 30 Hz input generator
    # measures capture/memory throughput, not random-number generation. Every
    # delivered frame also carries a tiny sequence marker so skipped frames
    # cannot accidentally hit the geometry cache. Account for both retained sources
    # and the next delivery buffer in the frame budget.
    source_pool = [pixels]
    if args.motion_input == "lsb-noise":
        rng = np.random.default_rng(20261015)
        source_pool.append(np.bitwise_xor(pixels, rng.integers(0, 2, pixels.shape, dtype=np.uint8)))
    source_epoch = workspace._source_epoch
    stop = threading.Event()
    generated = [0]
    producer_error = []
    completed = {}
    queue_wait_ms = {}
    gui_publish_ms = {}
    completion_lock = threading.Lock()
    original_compute = controller._compute

    def tracked_compute(intent):
        with completion_lock:
            queue_wait_ms[intent.id] = (time.perf_counter() - intent.submitted_at) * 1000
        try:
            return original_compute(intent)
        finally:
            with completion_lock:
                completed[intent.id] = time.perf_counter()

    controller._compute = tracked_compute
    original_accept = controller._accept

    def tracked_accept(intent, result):
        started = time.perf_counter()
        try:
            return original_accept(intent, result)
        finally:
            gui_publish_ms[intent.id] = (time.perf_counter() - started) * 1000

    controller._accept = tracked_accept

    def produce():
        due = time.perf_counter()
        try:
            while not stop.is_set():
                source = source_pool[generated[0] % len(source_pool)]
                owned = np.array(source, copy=True, order="C")
                if args.motion_input == "lsb-noise":
                    # Six-pixel cells survive the fixed threefold reduction.
                    # A 16-bit one-level marker makes every delivered frame
                    # distinct even when motion sampling skips an even count.
                    for bit in range(16):
                        owned[12:18, 12 + bit * 6:18 + bit * 6] = 200 + ((generated[0] >> bit) & 1)
                frame = FrameHandle.snapshot(owned, transfer_ownership=True,
                    source_id="qt-benchmark", sequence=generated[0],
                    metadata={"synthetic": True,
                              "acquisition_reserved_bytes": pixels.nbytes * (len(source_pool) + 1)})
                generated[0] += 1
                workspace._queue_frame(source_epoch, frame)
                due = max(due + 1 / args.frame_hz, time.perf_counter())
                stop.wait(max(0, due - time.perf_counter()))
        except Exception as exc:
            producer_error.append(str(exc))

    def accepted(record):
        trial = state["current_trial"]
        if trial is not None:
            trial["accepted_at"] = time.perf_counter()
            trial["record_id"] = record.id
            trial["width_um"] = record.diameter_um
            trial["controller_latency_ms"] = record.provenance["latency_ms"]
            trial["barrier_sequence"] = record.provenance.get("submission_barrier_sequence")

    rejected = []
    controller.recordAccepted.connect(accepted)
    def on_rejected(text):
        rejected.append(str(text))
        trial = state["current_trial"]
        if trial is not None:
            trial["rejected_at"] = time.perf_counter()
            trial["rejection_reason"] = str(text)
    controller.resultRejected.connect(on_rejected)
    producer = threading.Thread(target=produce, name="cu-ui-benchmark-source", daemon=True)
    producer.start()
    trials = []
    resources = None
    automatic = None
    automatic_active = None
    automatic_enabled = False
    automatic_results = []
    automatic_warmup_results = []
    automatic_phase = "warmup"
    automatic_timeline = {"phase_schema_version": 1, "clock": "time.perf_counter",
        "warmup_started_at": time.perf_counter(), "formal_started_at": None,
        "teardown_started_at": None, "close_requested_task_id": None}
    if args.background_automatic:
        from fdm.services.cu_automatic_worker import CuAutomaticWorker
        automatic = CuAutomaticWorker()

    def background():
        nonlocal automatic_active
        if automatic is None:
            return
        for response in automatic.poll():
            if automatic_active == response["task_id"]:
                automatic_results.append({"status": response["result"]["status"],
                    "candidates": len(response["result"].get("candidates", [])), "error": response.get("error"),
                    "task_id": response["task_id"], "phase": automatic_phase,
                    "worker_runtime": response.get("worker_runtime"),
                    "received_during_measurements": state["sample_start"] is not None and state["sample_end"] is None,
                    # Receipt on the GUI thread, not the worker's finish time.
                    "received_monotonic": time.perf_counter()})
                automatic_active = None
        if automatic_enabled and automatic_active is None:
            automatic_active = automatic.submit(pixels, algorithm=args.background_automatic)

    def wait_for(condition, seconds):
        return wait_until(condition, seconds, check=background)

    try:
        if not wait_for(lambda: controller.state == "ready" and controller.presented is not None, 10):
            raise RuntimeError("Synthetic live source failed to become ready")
        if automatic is not None:
            automatic_active = automatic.submit(pixels, algorithm=args.background_automatic)
            if not wait_for(lambda: automatic_active is None, 35):
                raise RuntimeError("Background worker warmup did not finish")
            automatic_warmup_results.extend(automatic_results)
            automatic_results.clear()
        with ResourceSampler() as resources:
            for index in range(args.warmup + args.samples):
                if index == args.warmup:
                    automatic_phase = "formal"
                    automatic_timeline["formal_started_at"] = time.perf_counter()
                    automatic_enabled = True
                    background()
                    state["sample_start"] = time.perf_counter()
                    state["generated_at_sample_start"] = generated[0]
                if not wait_for(lambda: controller.state == "ready", 5):
                    raise RuntimeError("Live field did not return to ready")
                trial = {"warmup": index < args.warmup, "first_click_at": time.perf_counter()}
                rejected_before = len(rejected)
                state["current_trial"] = trial
                token = controller.presented
                point = QPoint(round(start[0] * token.scale + token.offset[0]), round(start[1] * token.scale + token.offset[1]))
                QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point)
                wait_for(lambda: "draft_painted_at" in trial or "rejected_at" in trial, 2)
                if "draft_painted_at" not in trial:
                    trial["failure"] = "draft_rejected" if "rejected_at" in trial else "draft_not_painted"
                    controller.cancel_draft()
                else:
                    token = controller.presented
                    point = QPoint(round(end[0] * token.scale + token.offset[0]), round(end[1] * token.scale + token.offset[1]))
                    trial["second_click_at"] = time.perf_counter()
                    QTest.mouseClick(workspace.canvas, Qt.MouseButton.LeftButton, pos=point)
                    if controller._jobs:
                        trial["intent_id"] = controller._jobs[-1][0].id
                    wait_for(lambda: "record_painted_at" in trial or "rejected_at" in trial, 5)
                    if "record_painted_at" not in trial:
                        trial["failure"] = "measurement_rejected" if "rejected_at" in trial else "record_not_painted"
                    with completion_lock:
                        finished = completed.get(trial.get("intent_id"))
                        queued = queue_wait_ms.get(trial.get("intent_id"))
                    if finished is not None:
                        trial["computation_finished_at"] = finished
                    if queued is not None:
                        trial["queue_wait_ms"] = queued
                    if trial.get("intent_id") in gui_publish_ms:
                        trial["gui_publish_ms"] = gui_publish_ms[trial["intent_id"]]
                if "draft_painted_at" in trial:
                    trial["first_click_to_draft_paint_ms"] = (trial["draft_painted_at"] - trial["first_click_at"]) * 1000
                if "accepted_at" in trial:
                    trial["second_click_to_accept_ms"] = (trial["accepted_at"] - trial["second_click_at"]) * 1000
                if "record_painted_at" in trial:
                    trial["second_click_to_record_paint_ms"] = (trial["record_painted_at"] - trial["second_click_at"]) * 1000
                    trial["accept_to_record_paint_ms"] = (trial["record_painted_at"] - trial["accepted_at"]) * 1000
                if "computation_finished_at" in trial and "accepted_at" in trial:
                    trial["compute_done_to_gui_accept_ms"] = (trial["accepted_at"] - trial["computation_finished_at"]) * 1000
                if not trial["warmup"]:
                    trial["rejections"] = rejected[rejected_before:]
                    trials.append(trial)
                state["current_trial"] = None
                pause_end = time.perf_counter() + args.interval_ms / 1000
                wait_for(lambda: time.perf_counter() >= pause_end, args.interval_ms / 1000 + 1)
            automatic_enabled = False
            state["sample_end"] = time.perf_counter()
            automatic_timeline["measurement_ended_at"] = state["sample_end"]
            state["generated_at_sample_end"] = generated[0]
        if not wait_for(lambda: not controller._jobs and not controller.evidence.pending, 30):
            raise RuntimeError("Measurement/evidence queues did not drain")
        # Fast manual trials can finish before the full-frame task. Validate its
        # actual result without extending the measured latency/FPS window or
        # mistaking a teardown cancellation for a completed analysis. The worker
        # still owns its unchanged 30-second hard timeout.
        if automatic is not None and not wait_for(lambda: automatic_active is None, 35):
            raise RuntimeError("Background task did not return after its bounded deadline")
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not workspace.grab().save(str(args.screenshot)):
                raise OSError("Could not save rendered workspace screenshot")
        metrics = {name: distribution([trial[name] for trial in trials if name in trial]) for name in (
            "first_click_to_draft_paint_ms", "second_click_to_accept_ms", "second_click_to_record_paint_ms",
            "accept_to_record_paint_ms", "compute_done_to_gui_accept_ms", "controller_latency_ms", "queue_wait_ms", "gui_publish_ms")}
        screen = workspace.screen()
        display = {"window_size": [workspace.width(), workspace.height()],
            "canvas_size": [workspace.canvas.width(), workspace.canvas.height()],
            "device_pixel_ratio": workspace.devicePixelRatioF(),
            "logical_dpi": screen.logicalDotsPerInch() if screen else None,
            "screen_size": [screen.size().width(), screen.size().height()] if screen else None}
        return {"width": size, "shape": pixels.shape, "input_sha256": input_digest(pixels), "display": display,
            "save_images": save_images, "metrics": metrics, "trials": trials,
            "failures": dict(Counter(trial["failure"] for trial in trials if "failure" in trial)),
            "rejections": dict(Counter(rejected)), "producer_errors": producer_error,
            "frames_generated": generated[0], "distinct_frames_painted": len(state["painted_frame_ids"]),
            "motion_input": args.motion_input,
            "unique_sequence_marker": args.motion_input == "lsb-noise",
            "background_automatic": args.background_automatic,
            "background_results": automatic_results,
            "background_warmup_results": automatic_warmup_results,
            # Like background_results, this dict is finalized in finally.
            "background_timeline": automatic_timeline,
            "source_pool_sha256": [input_digest(source) for source in source_pool],
            "source_retained_bytes": sum(source.nbytes for source in source_pool),
            "source_generated_fps_during_measurements": (state["generated_at_sample_end"] - state["generated_at_sample_start"]) / (state["sample_end"] - state["sample_start"]),
            "measurement_window_seconds": state["sample_end"] - state["sample_start"],
            "distinct_frames_painted_during_measurements": len(state["sample_frame_ids"]),
            "painted_frames_per_second": len(state["sample_frame_ids"]) / (state["sample_end"] - state["sample_start"]),
            "frame_age_at_first_paint": distribution(state["frame_age_ms"]),
            "canvas_paint_count": state["paints"], "canvas_paint_duration": distribution(state["paint_ms"]),
            "evidence_states": dict(Counter((record.evidence or {}).get("state", "missing") for record in session.measurements.values())),
            "frame_budget_peak_bytes": controller.budget.peak_bytes,
            "motion_reduced_retained_bytes": controller._motion.retained_bytes,
            "motion_preparation_retained_bytes": controller._preparer.retained_bytes,
            "verify_before_present": controller.verify_before_present,
            "resources": resources.result,
            "boundary": "actual QWidget paint, Qt input and event loop; synthetic source, no physical camera or compositor presentation measurement"}
    finally:
        automatic_phase = "teardown"
        automatic_timeline["teardown_started_at"] = time.perf_counter()
        automatic_timeline["close_requested_task_id"] = automatic_active
        automatic_enabled = False
        if automatic is not None:
            automatic.close()
        stop.set()
        producer.join(5)
        wait_for(lambda: not controller.evidence.pending, 30)
        workspace.close()
        if not controller._closed:
            controller.close()
        host.close()
        QApplication.processEvents()
        evidence_directory.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--frame-hz", type=float, default=30)
    parser.add_argument("--interval-ms", type=float, default=100)
    parser.add_argument("--evidence", choices=("both", "off", "on"), default="both")
    parser.add_argument("--motion-input", choices=("stable", "lsb-noise"), default="stable")
    parser.add_argument("--background-automatic", choices=("traditional", "multiscale"),
        help="Keep one full-image worker task active during formal measurement samples.")
    parser.add_argument("--opencv-threads", type=int, default=None,
        help="Optional requested thread setting; omitted preserves the application's OpenCV default. GCD may not honor positive counts.")
    parser.add_argument("--window-width", type=int, default=1240)
    parser.add_argument("--window-height", type=int, default=840)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if min(args.sizes) < 320 or args.warmup < 0 or args.samples < 1 or args.frame_hz <= 0 or args.interval_ms < 0:
        parser.error("invalid size, sampling count or timing")
    application = QApplication.instance() or QApplication([])
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    build_environment = environment()
    build_environment["opencv_requested_threads"] = args.opencv_threads
    build_environment["opencv_reported_threads"] = cv2.getNumThreads()
    build_environment["opencv_parallel_framework"] = next((line.strip() for line in cv2.getBuildInformation().splitlines()
        if "Parallel framework:" in line), "unknown")
    for relative in ("src/fdm/ui/live_experiment_workspace.py", "scripts/benchmark_cu_live_ui.py",
                     "scripts/benchmark_cu_live_measurement.py"):
        build_environment["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    report = {"schema_version": 1, "environment": build_environment, "qt_platform": application.platformName(),
              "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saving = (False, True) if args.evidence == "both" else (args.evidence == "on",)
    for size in args.sizes:
        for save_images in saving:
            print(f"Qt {size}, PNG={'on' if save_images else 'off'}", flush=True)
            report["cases"].append(run_case(args, size, save_images))
            temporary = args.output.with_suffix(args.output.suffix + ".pending")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
            os.replace(temporary, args.output)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
