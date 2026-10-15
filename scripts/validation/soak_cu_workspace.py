#!/usr/bin/env python3
"""Real-workspace CU endurance run: Qt painting, two clicks, PNG and automatic IO.

Run alone for acceptance; the default is one hour and at least 10,000 accepted
measurements. No Qt platform is forced: Windows uses its native window system.
For a short harness check, use --duration-seconds 5 --minimum-accepted 1
--motion-interval-seconds 1 --qt-platform offscreen. A short run is never labelled
as the one-hour acceptance. Synthetic pixels do not validate real camera input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    # Import only after the explicit platform override; otherwise preserve the
    # installed application's native Qt selection and OpenCV thread default.
    if args.qt_platform:
        os.environ["QT_QPA_PLATFORM"] = args.qt_platform
    from scripts.benchmark_cu_live_measurement import BoundedSamples, ResourceSampler, environment, fixture, input_digest
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
    from fdm.services.cu_automatic_worker import CuAutomaticWorker
    import fdm.ui.live_experiment_workspace as workspace_module

    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    app = QApplication.instance() or QApplication([])
    build_environment = environment()
    build_environment["opencv_requested_threads"] = args.opencv_threads
    build_environment["qt_platform"] = app.platformName()
    build_environment["parallel_framework"] = [line.strip() for line in cv2.getBuildInformation().splitlines()
                                                 if "Parallel framework" in line]
    import hashlib
    for name in ("scripts/validation/soak_cu_workspace.py", "src/fdm/ui/live_experiment_workspace.py"):
        build_environment["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

    metric_names = ("first_click_to_draft_paint", "second_click_to_accept", "second_click_to_record_paint",
                    "accept_to_record_paint", "compute_done_to_accept", "canvas_paint_duration",
                    "gui_publish_ms", "gui_publish_failed_ms",
                    "frame_age_at_first_paint", "event_loop_gap", "source_snapshot_and_handoff",
                    "scene_input_to_first_paint", "scene_input_to_motion_complete", "motion_complete_to_clear",
                    "scene_input_to_clear", "scene_input_to_clear_paint", "clear_to_clear_paint")
    metrics = {name: BoundedSamples() for name in metric_names}
    quarters = [{name: BoundedSamples() for name in (*metric_names[:3], "gui_publish_ms")}
                for _ in range(max(1, math.ceil(args.duration_seconds / 900)))]
    state = {"phase": "startup", "started": None, "gesture": None, "publishing_intent": None,
             "paint_count": 0, "distinct_paints": 0, "last_painted_sequence": -1,
             "attempt_slots": 0, "first_clicks": 0, "submitted": 0, "accepted": 0,
             "records_painted": 0, "generated": 0, "field_changes": 0,
             "old_epoch_accepted": 0, "old_scene_accepted": 0, "unmapped_accepted": 0,
             "wrong_width": 0, "expected_motion_cancelled": 0, "rejected_without_motion": 0,
             "submitted_motion_cancelled": 0, "submitted_rejected_other": 0,
             "draft_motion_cancelled": 0, "draft_lost_without_submission": 0,
             "accepted_unpainted_due_motion": 0, "accepted_unpainted_other": 0,
             "transition_inputs": 0, "transition_clear_events": 0, "transition_completed": 0,
             "unexpected_field_changes": 0, "fatal": False}
    pending, awaiting_paint, completed = {}, {}, {}
    completion_lock = threading.Lock()
    transition_lock = threading.Lock()
    transitions, transition_examples = {}, []
    last_unstable_motion = [None]
    errors, rejected, expected_rejections, statuses, skipped, automatic_results = (Counter() for _ in range(6))
    queue_max, violation_examples, exception_examples = Counter(), [], []
    automatic_latency = BoundedSamples()

    def latency(name, value, trial=None):
        metrics[name].append(value)
        if trial is not None and name in quarters[0]:
            index = min(len(quarters) - 1, int(max(0, trial["first_at"] - state["started"]) // 900))
            quarters[index][name].append(value)

    def finish_transition_telemetry():
        # Caller holds transition_lock. Only primitive timings/IDs are retained;
        # never retain the frame used by the detector or its executor holder.
        for identifier, event in list(transitions.items()):
            if all(name in event for name in ("first_painted_at", "cleared_at", "clear_painted_at")):
                state["transition_completed"] += 1
                if len(transition_examples) < 20:
                    transition_examples.append({**event, "transition_id": identifier})
                del transitions[identifier]

    original_canvas = workspace_module.LiveMeasurementCanvas

    class MeasuredCanvas(original_canvas):
        def paintEvent(self, event):
            began = time.perf_counter()
            super().paintEvent(event)  # actual QPainter and real presented token
            now = time.perf_counter()
            if state["phase"] == "run":
                state["paint_count"] += 1
                latency("canvas_paint_duration", (now - began) * 1000)
            if self.frame is not None and self.frame.sequence != state["last_painted_sequence"]:
                if self.frame.sequence < state["last_painted_sequence"]:
                    errors["paint_sequence_regressed"] += 1
                state["last_painted_sequence"] = self.frame.sequence
                if state["phase"] == "run" and self.frame.received_at >= state["started"]:
                    state["distinct_paints"] += 1
                    latency("frame_age_at_first_paint", (now - self.frame.received_at) * 1000)
            if self.frame is not None:
                if any(self.controller.session.measurements[identifier].field_id != self.controller.field_id
                       for identifier in self.controller.current_records):
                    errors["old_field_overlay_painted"] += 1
                identifier = self.frame.metadata.get("synthetic_transition_id", 0)
                with transition_lock:
                    event = transitions.get(identifier)
                    if event is not None and "first_painted_at" not in event:
                        event["first_painted_at"] = now
                        latency("scene_input_to_first_paint", (now - event["input_at"]) * 1000)
                    for event in transitions.values():
                        if ("cleared_at" in event and "clear_painted_at" not in event
                                and self.controller.field_epoch >= event["cleared_epoch"]):
                            event["clear_painted_at"] = now
                            latency("scene_input_to_clear_paint", (now - event["input_at"]) * 1000)
                            latency("clear_to_clear_paint", (now - event["cleared_at"]) * 1000)
                    finish_transition_telemetry()
            trial = state["gesture"]
            if trial is not None and self.controller.draft and "draft_at" not in trial:
                trial["draft_at"] = now
                latency("first_click_to_draft_paint", (now - trial["first_at"]) * 1000, trial)
            if awaiting_paint:
                visible = set(self.controller.current_records)
                for record_id, trial in list(awaiting_paint.items()):
                    if record_id in visible:
                        latency("second_click_to_record_paint", (now - trial["second_at"]) * 1000, trial)
                        latency("accept_to_record_paint", (now - trial["accepted_at"]) * 1000)
                        state["records_painted"] += 1
                        del awaiting_paint[record_id]

    class Host(QMainWindow):
        def __init__(self, session):
            super().__init__()
            self.project = ProjectState.empty()
            self.project.live_experiments.append(session)
            self._images = {}
            self.save_project_action = QAction(self)

        def current_document(self):
            return None

    session = ExperimentSession(save_images=args.save_images, sample_name="Synthetic real-Qt one-hour soak")
    session.add_operator("soak")
    session.set_materials(default_materials(), direction="warp")
    session.set_materials(default_materials(), direction="weft")
    session.establish_history_baseline()
    host = Host(session)
    workspace_module.LiveMeasurementCanvas = MeasuredCanvas
    try:
        view = workspace_module.open_live_experiment(host)
    finally:
        workspace_module.LiveMeasurementCanvas = original_canvas
    controller = view.controller
    evidence_directory = args.evidence_dir or args.output.with_suffix("").with_name(args.output.stem + "-evidence")
    evidence_directory.mkdir(parents=True, exist_ok=True)
    if any(evidence_directory.iterdir()):
        view.close()
        host.close()
        raise ValueError("Evidence directory must be empty; use a new output path for each run")
    old_directory = controller.evidence.directory
    controller.evidence.directory = evidence_directory
    old_directory.rmdir()
    controller.set_calibration(CalibrationSnapshot(1, 1, 1, "synthetic soak"))
    view.resize(args.window_width, args.window_height)
    view.activateWindow()
    # A single long horizontal stripe shifted along its own axis by 12 pixels
    # falls below the recovered CU motion gates. Use the existing sparse-stripe
    # fixture and move across the stripes: real image changes, unchanged rules.
    pixels, _, _ = fixture(args.size, kind="sparse")
    stripe_row = int(np.linspace(pixels.shape[0] // 8, pixels.shape[0] * 7 // 8, 8).astype(int)[3])
    start, end = (args.size // 2, stripe_row - 50), (args.size // 2, stripe_row + 50)
    moved = np.roll(pixels, 12, axis=0)
    moved.setflags(write=False)
    source_pool = (pixels, moved)
    from dataclasses import asdict
    from fdm.live_experiment.motion import FieldMotionDetector, reduced_gray
    motion_preflight = []
    for first, second in ((pixels, moved), (moved, pixels)):
        detector = FieldMotionDetector()
        key = ("preflight", 0, (args.size, pixels.shape[0]))
        detector.set_reference_reduced(reduced_gray(first), key)
        result = detector.compare_reduced(reduced_gray(second), key, "preflight:1", 1)
        motion_preflight.append(asdict(result))
    del detector, result, first, second
    if any(result["stable"] for result in motion_preflight):
        view.close()
        host.close()
        raise ValueError("The fixed 12-pixel motion fixture does not trigger the production motion detector")
    source_epoch = view._source_epoch
    stop = threading.Event()
    automatic = CuAutomaticWorker()
    automatic_active = None
    original_compute, original_finish, original_accept = controller._compute, controller.finish, controller._accept
    original_motion_work = controller._motion_work

    def tracked_motion_work(holder, reset):
        # Copy only metadata scalars before production clears holder and releases
        # its full-frame lease. A local `frame = holder[0]` would falsify memory.
        identifier = holder[0].metadata.get("synthetic_transition_id", 0)
        sequence = holder[0].sequence
        result = original_motion_work(holder, reset)
        if not result.stable:
            with transition_lock:
                last_unstable_motion[0] = {"transition_id": identifier, "motion_sequence": sequence,
                    "motion_completed_at": time.perf_counter(), "motion_reason": result.reason}
        return result

    def tracked_compute(intent):
        try:
            return original_compute(intent)
        finally:
            with completion_lock:
                completed[intent.id] = time.perf_counter()

    def tracked_finish(*arguments, **keywords):
        identifier = original_finish(*arguments, **keywords)
        trial = state["gesture"]
        if trial is not None:
            pending[identifier] = trial
            trial["intent_id"] = identifier
            state["submitted"] += 1
        return identifier

    def tracked_accept(intent, result):
        state["publishing_intent"] = intent
        trial = pending.get(intent.id)
        started = time.perf_counter()
        succeeded = False
        try:
            returned = original_accept(intent, result)
            succeeded = True
            return returned
        finally:
            # Includes synchronous UI signal handlers; the later QPainter wait
            # remains measured separately by accept_to_record_paint.
            elapsed_ms = (time.perf_counter() - started) * 1000
            latency("gui_publish_ms", elapsed_ms, trial)
            if not succeeded:
                metrics["gui_publish_failed_ms"].append(elapsed_ms)
            state["publishing_intent"] = None

    controller._compute, controller.finish, controller._accept = tracked_compute, tracked_finish, tracked_accept
    controller._motion_work = tracked_motion_work

    def on_accepted(record):
        now = time.perf_counter()
        state["accepted"] += 1
        intent = state["publishing_intent"]
        trial = pending.pop(intent.id, None) if intent is not None else None
        if trial is None:
            state["unmapped_accepted"] += 1
            return
        token = controller.presented
        current_scene = token.frame.metadata.get("synthetic_scene") if token is not None else None
        record_scene = record.provenance.get("frame_metadata", {}).get("synthetic_scene")
        bad_epoch = (trial["field_epoch"] != controller.field_epoch or trial["field_id"] != controller.field_id
                     or intent.token.field_epoch != trial["field_epoch"] or record.field_id != trial["field_id"])
        bad_scene = record_scene != current_scene or record_scene != trial["scene"]
        state["old_epoch_accepted"] += int(bad_epoch)
        state["old_scene_accepted"] += int(bad_scene)
        state["wrong_width"] += int(abs(record.diameter_um - 19.0) > 1e-9)
        if (bad_epoch or bad_scene) and len(violation_examples) < 20:
            violation_examples.append({"record_id": record.id, "record_scene": record_scene,
                "presented_scene": current_scene, "first_click_scene": trial["scene"],
                "first_click_epoch": trial["field_epoch"], "accept_epoch": controller.field_epoch,
                "first_click_field": trial["field_id"], "record_field": record.field_id})
        trial["accepted_at"] = now
        latency("second_click_to_accept", (now - trial["second_at"]) * 1000, trial)
        with completion_lock:
            done = completed.pop(intent.id, None)
        if done is not None:
            latency("compute_done_to_accept", (now - done) * 1000)
        awaiting_paint[record.id] = trial

    def on_field_changed(_epoch, reason):
        is_motion = reason == "视场已变化，请重新测量"
        if state["phase"] in ("run", "drain"):
            if is_motion:
                state["field_changes"] += 1
                now = time.perf_counter()
                with transition_lock:
                    motion, last_unstable_motion[0] = last_unstable_motion[0], None
                    event = transitions.get(motion["transition_id"]) if motion else None
                    if event is None or "cleared_at" in event:
                        errors["motion_clear_without_matching_input_transition"] += 1
                    else:
                        event.update(motion, cleared_at=now, cleared_epoch=_epoch)
                        state["transition_clear_events"] += 1
                        latency("scene_input_to_motion_complete", (motion["motion_completed_at"] - event["input_at"]) * 1000)
                        latency("motion_complete_to_clear", (now - motion["motion_completed_at"]) * 1000)
                        latency("scene_input_to_clear", (now - event["input_at"]) * 1000)
            else:
                state["unexpected_field_changes"] += 1
        for trial in pending.values():
            trial["motion_cancelled"] = trial.get("motion_cancelled", False) or is_motion
        if state["gesture"] is not None:
            state["gesture"]["motion_cancelled"] = is_motion
        state["accepted_unpainted_due_motion" if is_motion else "accepted_unpainted_other"] += len(awaiting_paint)
        awaiting_paint.clear()

    def on_rejected(message):
        text = str(message)
        (expected_rejections if "视场已变化" in text or "已取消" in text else rejected)[text] += 1

    controller.recordAccepted.connect(on_accepted)
    controller.fieldChanged.connect(on_field_changed)
    controller.resultRejected.connect(on_rejected)
    controller.statusChanged.connect(lambda message: statuses.update([str(message)]))
    previous_hook = sys.excepthook

    def exception_hook(kind, value, tb):
        errors["qt_callback_exception"] += 1
        state["fatal"] = True
        if len(exception_examples) < 10:
            exception_examples.append("".join(traceback.format_exception(kind, value, tb)))
        previous_hook(kind, value, tb)

    sys.excepthook = exception_hook

    def produce():
        due = time.perf_counter()
        previous_transition = 0
        try:
            while not stop.is_set():
                now = time.perf_counter()
                elapsed = max(0, now - state["started"]) if state["started"] is not None else 0
                transition_id = int(elapsed // args.motion_interval_seconds)
                scene = transition_id % 2
                if transition_id != previous_transition:
                    with transition_lock:
                        state["transition_inputs"] += 1
                        if len(transitions) >= 1024:
                            raise RuntimeError("Unresolved transition telemetry exceeded 1024 events")
                        transitions[transition_id] = {"input_at": now, "input_sequence": state["generated"], "scene": scene}
                    previous_transition = transition_id
                frame = FrameHandle.snapshot(source_pool[scene], source_id="real-qt-soak", sequence=state["generated"],
                    received_at=now, metadata={"synthetic": True, "synthetic_scene": scene,
                        "synthetic_transition_id": transition_id,
                        "acquisition_reserved_bytes": pixels.nbytes * 3})
                state["generated"] += 1
                view._queue_frame(source_epoch, frame)
                del frame
                if state["phase"] == "run":
                    latency("source_snapshot_and_handoff", (time.perf_counter() - now) * 1000)
                due = max(due + 1 / args.frame_hz, time.perf_counter())
                stop.wait(max(0, due - time.perf_counter()))
        except Exception as exc:
            errors["producer: " + str(exc)] += 1
            state["fatal"] = True

    producer = threading.Thread(target=produce, name="cu-qt-soak-source", daemon=True)

    def queue_observations():
        values = {"manual_jobs": len(controller._jobs), "pending_classification": len(controller.pending_classification),
            "confirmation": int(controller._confirmation is not None), "evidence_writes": len(controller.evidence._futures),
            "mailbox": int(view._frame_mailbox is not None), "automatic_inflight": int(automatic.busy),
            "tracked_intents": len(pending), "awaiting_record_paint": len(awaiting_paint)}
        for key, value in values.items():
            queue_max[key] = max(queue_max[key], value)
        active_ids = {intent.id for intent, _ in controller._jobs}
        active_ids.update(intent.id for intent, _ in controller.pending_classification)
        if controller._confirmation:
            active_ids.update(intent.id for intent, _ in controller._confirmation["intents"])
        for identifier, trial in list(pending.items()):
            if identifier not in active_ids:
                state["expected_motion_cancelled" if trial.get("motion_cancelled") else "rejected_without_motion"] += 1
                state["submitted_motion_cancelled" if trial.get("motion_cancelled") else "submitted_rejected_other"] += 1
                del pending[identifier]
                with completion_lock:
                    completed.pop(identifier, None)

    def poll_automatic():
        nonlocal automatic_active
        for response in automatic.poll():
            result = response.get("result") or {}
            automatic_results[result.get("status", "error")] += 1
            if automatic_active is not None:
                automatic_latency.append((time.perf_counter() - automatic_active) * 1000)
            if result.get("status") not in {"ok", "success", "empty"}:
                errors["automatic: " + str(result.get("reason") or response.get("error"))] += 1
            automatic_active = None
            # Candidates are intentionally discarded here, never added to the
            # session or canvas. This worker exists solely for concurrent load.

    def click(image_point):
        token = controller.presented
        point = QPoint(round(image_point[0] * token.scale + token.offset[0]),
                       round(image_point[1] * token.scale + token.offset[1]))
        QTest.mouseClick(view.canvas, Qt.MouseButton.LeftButton, pos=point, delay=0)

    def gesture_step(now):
        trial = state["gesture"]
        if trial is None:
            return
        if trial.get("motion_cancelled") or controller.draft is None:
            state["expected_motion_cancelled" if trial.get("motion_cancelled") else "rejected_without_motion"] += 1
            state["draft_motion_cancelled" if trial.get("motion_cancelled") else "draft_lost_without_submission"] += 1
            state["gesture"] = None
        elif "draft_at" in trial:
            trial["second_at"] = time.perf_counter()
            click(trial["end"])
            if "intent_id" not in trial:
                skipped["second_click_not_submitted"] += 1
                controller.cancel_draft()
            state["gesture"] = None
        elif now - trial["first_at"] > 2:
            errors["draft_not_painted_within_two_seconds"] += 1
            controller.cancel_draft()
            state["gesture"] = None

    report = {"schema_version": 1, "environment": build_environment,
              "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "limits": ["Actual QTest input, QWidget/QPainter and event loop; physical display scan-out is not measured.",
                         "Existing sparse-stripe fixture and its vertical 12-pixel translation; no camera/CU hardware acceptance.",
                         "Frame budget includes two retained source arrays and next delivery; algorithm/Qt/PNG RSS reported separately.",
                         "Metric reservoirs are bounded; the production ledger and undo history intentionally retain accepted records.",
                         "Resource sampling covers active work and queue drain; post-run PNG verification and ledger reload are separate."]}
    producer.start()
    run_seconds = drain_seconds = 0.0
    unfinished = True
    try:
        ready_deadline = time.perf_counter() + 15
        while time.perf_counter() < ready_deadline and not state["fatal"]:
            app.processEvents()
            if controller.state == "ready" and controller.presented is not None:
                break
            time.sleep(.001)
        else:
            raise RuntimeError("Synthetic source failed to reach a painted, stable state")
        state["started"] = started = time.perf_counter()
        from datetime import datetime, timezone
        report["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["phase"] = "run"
        generated_at_start = state["generated"]
        measure_due = automatic_due = last_tick = started
        progress_due = started + 60
        with ResourceSampler() as resources:
            while time.perf_counter() - started < args.duration_seconds and not state["fatal"]:
                now = time.perf_counter()
                latency("event_loop_gap", (now - last_tick) * 1000)
                last_tick = now
                app.processEvents()
                poll_automatic()
                queue_observations()
                gesture_step(now)
                if now >= measure_due:
                    state["attempt_slots"] += 1
                    measure_due = now + 1 / args.measurement_hz
                    if state["gesture"] is not None:
                        skipped["gesture_still_active"] += 1
                    elif controller.state != "ready" or controller.presented is None:
                        skipped["field_not_ready"] += 1
                    elif len(controller._jobs) >= 16:
                        skipped["manual_queue_full"] += 1
                    else:
                        token = controller.presented
                        scene = token.frame.metadata.get("synthetic_scene")
                        offset = 12 * int(scene or 0)
                        state["gesture"] = {"first_at": time.perf_counter(), "field_epoch": token.field_epoch,
                            "field_id": controller.field_id, "scene": scene,
                            "first_frame_id": token.frame.frame_id,
                            "start": (start[0], start[1] + offset), "end": (end[0], end[1] + offset)}
                        state["first_clicks"] += 1
                        click(state["gesture"]["start"])
                        if controller.draft is None:
                            skipped["first_click_not_started"] += 1
                            state["gesture"] = None
                if now >= automatic_due:
                    automatic_due = now + 10
                    if automatic.busy:
                        skipped["automatic_busy_at_ten_second_slot"] += 1
                    else:
                        try:
                            scene = int((now - started) // args.motion_interval_seconds) % 2
                            automatic.submit(source_pool[scene], "traditional")
                            automatic_active = time.perf_counter()
                        except Exception as exc:
                            errors["automatic_submit: " + str(exc)] += 1
                if now >= progress_due:
                    progress_due = now + 60
                    progress = {"elapsed_seconds": now - started, "accepted": state["accepted"],
                        "distinct_painted_fps": state["distinct_paints"] / (now - started),
                        "evidence_pending": len(controller.evidence._futures), "frame_budget_bytes": controller.budget.used_bytes,
                        "old_epoch_accepted": state["old_epoch_accepted"], "old_scene_accepted": state["old_scene_accepted"]}
                    print(json.dumps({"progress": progress}, allow_nan=False), flush=True)
                    write_report(args.output.with_suffix(args.output.suffix + ".progress"), progress)
                time.sleep(.001)
            run_seconds = time.perf_counter() - started
            generated_during_run = state["generated"] - generated_at_start
            state["phase"] = "drain"
            stop.set()
            producer.join(5)
            if producer.is_alive():
                errors["producer_failed_to_stop"] += 1
            if state["gesture"] is not None:
                skipped["draft_cancelled_at_run_end"] += 1
                state["gesture"] = None
                controller.cancel_draft()
            drain_start = time.perf_counter()
            while time.perf_counter() - drain_start < args.drain_timeout_seconds:
                app.processEvents()
                poll_automatic()
                queue_observations()
                if not (controller._jobs or controller.evidence.pending or controller._motion_future
                        or automatic.busy or awaiting_paint or transitions):
                    break
                time.sleep(.002)
            drain_seconds = time.perf_counter() - drain_start
            unfinished = bool(controller._jobs or controller.evidence.pending or controller._motion_future
                              or automatic.busy or awaiting_paint or transitions)
        screen = view.screen()
        report["display"] = {"window_size": [view.width(), view.height()],
            "canvas_size": [view.canvas.width(), view.canvas.height()],
            "device_pixel_ratio": view.devicePixelRatioF(),
            "logical_dpi": screen.logicalDotsPerInch() if screen else None,
            "screen_size": [screen.size().width(), screen.size().height()] if screen else None}
        report["resources"] = resources.result
        report["generated_during_run"] = generated_during_run
        report["frame_budget_peak_bytes"] = controller.budget.peak_bytes
        report["motion_retained_bytes"] = controller._motion.retained_bytes
    except (Exception, KeyboardInterrupt) as exc:
        errors["run: " + str(exc)] += 1
        if len(exception_examples) < 10:
            exception_examples.append(traceback.format_exc())
    finally:
        state["phase"] = "cleanup"
        stop.set()
        producer.join(5)
        # Finish accepted PNG writes before the window is allowed to close.
        # This may exceed the diagnostic drain deadline; it never abandons a
        # confirmed record's write merely to return a successful report.
        for future in list(controller.evidence._futures.values()):
            try:
                future.result()
            except Exception as exc:
                errors["evidence_cleanup: " + str(exc)] += 1
        controller.poll()
        automatic.close()
        view.close()
        if not controller._closed:
            errors["workspace_close_not_completed"] += 1
            controller.close()
        host.close()
        app.processEvents()
        sys.excepthook = previous_hook
    saved_states = Counter((record.evidence or {}).get("state", "missing") for record in session.measurements.values())
    files = list(evidence_directory.glob("*.png"))
    verification_started = time.perf_counter()
    evidence_verification = {"unique_files_checked": 0, "unique_png_payloads_decoded": 0,
        "missing_or_wrong_hash": 0, "wrong_pixels_or_dimensions": 0, "extra_files": 0}
    expected_files, decoded_payloads = {}, {}
    if args.save_images:
        for record in session.measurements.values():
            entry = record.evidence or {}
            if entry.get("state") == "saved":
                expected_files[Path(entry["path"]).resolve()] = (entry, record.provenance["frame_metadata"].get("synthetic_scene"))
        evidence_verification["extra_files"] = len({path.resolve() for path in files} - expected_files.keys())
        for path, (entry, scene) in expected_files.items():
            try:
                encoded = path.read_bytes()
                digest = hashlib.sha256(encoded).hexdigest()
                evidence_verification["unique_files_checked"] += 1
                if digest != entry.get("sha256"):
                    evidence_verification["missing_or_wrong_hash"] += 1
                    continue
                # Every file is hashed. The two fixed source states produce the
                # same PNG bytes for equal pixels; decode each distinct payload
                # once, and still verify its scene assignment on every record.
                if digest not in decoded_payloads:
                    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
                    decoded_payloads[digest] = {index for index, source in enumerate(source_pool)
                        if decoded is not None and decoded.shape == source.shape and np.array_equal(decoded, source)}
                    evidence_verification["unique_png_payloads_decoded"] += 1
                    del decoded
                if scene not in decoded_payloads[digest]:
                    evidence_verification["wrong_pixels_or_dimensions"] += 1
            except (OSError, ValueError, cv2.error):
                evidence_verification["missing_or_wrong_hash"] += 1
    evidence_verification["seconds"] = time.perf_counter() - verification_started
    snapshot_path = args.output.with_suffix("").with_name(args.output.stem + "-ledger.json")
    snapshot_verified = False
    try:
        payload = session.to_dict()
        write_report(snapshot_path, payload)
        persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))
        restored = ExperimentSession.from_dict(persisted)
        # JSON represents tuple coordinates/diagnostics as lists. Compare the
        # exact JSON values, not Python's incidental tuple-versus-list types.
        snapshot_verified = json.loads(json.dumps(restored.to_dict(), allow_nan=False)) == persisted
        del restored, payload, persisted
    except Exception as exc:
        errors["ledger_snapshot: " + str(exc)] += 1
    all_fields_valid = all(record.field_id in session.fields and
        session.fields[record.field_id].get("source_id") == "real-qt-soak" for record in session.measurements.values())
    checks = {"requested_duration_completed": run_seconds >= args.duration_seconds,
        "minimum_accepted_reached": len(session.measurements) >= args.minimum_accepted,
        "all_records_have_valid_field": all_fields_valid,
        "old_epoch_accepted_zero": state["old_epoch_accepted"] == 0,
        "old_scene_accepted_zero": state["old_scene_accepted"] == 0,
        "accepted_event_count_matches_ledger": state["accepted"] == len(session.measurements),
        "all_accepted_mapped_to_click_intent": state["unmapped_accepted"] == 0,
        "submitted_intents_accounted_for": (not pending and state["submitted"] ==
            state["accepted"] + state["submitted_motion_cancelled"] + state["submitted_rejected_other"]),
        "accepted_paint_outcomes_accounted_for": (state["accepted"] == state["records_painted"] +
            state["accepted_unpainted_due_motion"] + state["accepted_unpainted_other"] and not awaiting_paint),
        "undo_history_preserves_each_measurement": len(session._undo) == len(session.measurements),
        "ledger_snapshot_roundtrip_exact": snapshot_verified,
        "background_automatic_executed": sum(automatic_results.values()) > 0,
        "no_automatic_candidates_entered_ledger": all(record.measurement_method == "assisted" for record in session.measurements.values()),
        "known_width_unchanged": state["wrong_width"] == 0,
        "evidence_complete": (saved_states.get("saved", 0) == len(session.measurements) if args.save_images else
                              saved_states.get("disabled", 0) == len(session.measurements)),
        "evidence_files_exist_and_match_pixels": not any(evidence_verification[key] for key in
            ("missing_or_wrong_hash", "wrong_pixels_or_dimensions", "extra_files")),
        "queues_drained_before_deadline": not unfinished,
        "frame_budget_within_512_mib": controller.budget.peak_bytes <= 512 * 1024**2,
        "all_frame_leases_released": controller.budget.used_bytes == 0,
        "producer_stopped": not producer.is_alive(), "unexpected_errors_zero": not errors,
        "scheduled_motion_detected": (state["transition_inputs"] >= max(0, int((run_seconds - 1 / args.frame_hz) // args.motion_interval_seconds))
            and state["transition_inputs"] == state["transition_clear_events"] == state["transition_completed"] and not transitions),
        "unexpected_field_changes_zero": state["unexpected_field_changes"] == 0,
        "unexpected_rejections_zero": not rejected and state["rejected_without_motion"] == 0}
    report.update(requested_seconds=args.duration_seconds, active_seconds=run_seconds, drain_seconds=drain_seconds,
        formal_one_hour_target=(args.duration_seconds >= 3600 and args.minimum_accepted >= 10000),
        shape=list(pixels.shape), fixture_kind="sparse", translation_pixels=[0, 12], motion_preflight=motion_preflight,
        source_pool_sha256=[input_digest(array) for array in source_pool],
        source_retained_bytes=sum(array.nbytes for array in source_pool), acquisition_reserved_bytes=pixels.nbytes * 3,
        distinct_painted_fps=state["distinct_paints"] / run_seconds if run_seconds else 0,
        generated_fps=report.get("generated_during_run", 0) / run_seconds if run_seconds else 0,
        counters={key: value for key, value in state.items() if key not in {"gesture", "publishing_intent", "started"}},
        metrics={name: values.report() for name, values in metrics.items()},
        latency_by_fifteen_minutes=[{"start_seconds": index * 900, "end_seconds": min((index + 1) * 900, args.duration_seconds),
            "metrics": {name: values.report() for name, values in group.items()}} for index, group in enumerate(quarters)],
        queue_max=dict(queue_max), history_count=len(session.measurements), undo_history_count=len(session._undo),
        field_count=len(session.fields), evidence_states=dict(saved_states), evidence_file_count=len(files),
        evidence_bytes=sum(path.stat().st_size for path in files), evidence_directory=str(evidence_directory.resolve()),
        evidence_verification=evidence_verification, ledger_snapshot=str(snapshot_path.resolve()),
        ledger_snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest() if snapshot_path.is_file() else None,
        motion_transition_examples=transition_examples, unresolved_motion_transitions=list(transitions.values()),
        errors=dict(errors), unexpected_rejections=dict(rejected), expected_motion_rejections=dict(expected_rejections),
        skipped_attempts=dict(skipped), status_counts=dict(statuses), violation_examples=violation_examples,
        exception_examples=exception_examples, background_automatic_results=dict(automatic_results),
        background_automatic_latency=automatic_latency.report(),
        automatic_candidates_entered_ledger=sum(record.measurement_method.startswith("automatic") for record in session.measurements.values()),
        checks=checks, passed=all(checks.values()), frame_budget_after_close=controller.budget.used_bytes)
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"],
        "active_seconds": run_seconds, "accepted": len(session.measurements), "checks": checks}, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=float, default=3600)
    parser.add_argument("--minimum-accepted", type=int, default=10000)
    parser.add_argument("--frame-hz", type=float, default=30)
    parser.add_argument("--measurement-hz", type=float, default=4)
    parser.add_argument("--motion-interval-seconds", type=float, default=10)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--drain-timeout-seconds", type=float, default=60)
    parser.add_argument("--window-width", type=int, default=1240)
    parser.add_argument("--window-height", type=int, default=840)
    parser.add_argument("--qt-platform", help="Explicit Qt override; omitted preserves native Windows/macOS selection")
    parser.add_argument("--opencv-threads", type=int, help="Optional request; omitted preserves production default")
    args = parser.parse_args()
    if min(args.duration_seconds, args.frame_hz, args.measurement_hz, args.motion_interval_seconds,
           args.drain_timeout_seconds) <= 0 or args.size < 320 or args.minimum_accepted < 0:
        parser.error("Positive timings/rates, size >= 320 and minimum-accepted >= 0 are required")
    return run(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
