"""Real Qt event-loop comparison of owned-image collection and trial interaction.

python scripts/validation/benchmark_trial_cleanup.py --output build/trial-cleanup-benchmark/result.json

The workload uses a real LiveExperimentController and LiveMeasurementCanvas with
a deterministic 30 Hz simulated camera, real motion verification, asynchronous
free-line measurement, record acceptance and paint events. It deliberately omits
Excel and the trial sidebar. No sleep or QApplication.processEvents appears in
the measured workload; the next measurement is queued immediately after the last
acceptance. Cleanup's own production batching/yields are unchanged.

The disposable image fixtures are created BEFORE measurement by a child process
through the actual ownership journal. It terminates without the normal lifecycle
cleanup; the operating system releases its instance lock. The benchmark cleans
only its private output directory, never the user's application cache.
"""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/trial-cleanup-benchmark/result.json")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--files", type=int, default=8192)
    parser.add_argument("--frame-ms", type=int, default=33)
    parser.add_argument("--seed-child", type=Path, default=None)
    return parser.parse_args()


def seed_child(root, count):
    from fdm.live_experiment.evidence_resources import EvidenceResources
    manager = EvidenceResources(root)
    class Store:
        pass
    store = Store()
    key, directory = manager.register_store(store)
    # A complete tiny PNG is enough: collector validates ownership/inode, not
    # image pixels. Each claim is issued by the unmodified production writer.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO2kAAAAASUVORK5CYII=")
    for index in range(count):
        path = directory / (hashlib.sha256(f"fixture:{index}".encode()).hexdigest() + ".png")
        with manager.open_pending(key, path) as handle:
            handle.write(png)
        os.replace(path.with_suffix(".pending"), path)
    print(json.dumps({"pid": os.getpid(), "directory": str(directory), "files": count}), flush=True)
    os._exit(0)  # Deliberate dead-instance fixture; never deletes unrelated data.


def distribution(values):
    import numpy as np
    data = np.asarray(values, dtype=float)
    return {"count": len(values), "p50_ms": float(np.percentile(data, 50)) if len(data) else None,
            "p95_ms": float(np.percentile(data, 95)) if len(data) else None,
            "max_ms": float(np.max(data)) if len(data) else None}


def main(args):
    import numpy as np
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
    from PySide6 import __version__ as qt_binding_version
    from PySide6.QtCore import QObject, QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from fdm.live_experiment import ExperimentSession, default_materials, evidence_resources
    from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
    from fdm.live_experiment.evidence_resources import EvidenceResources
    from fdm.live_experiment.frames import FrameHandle
    from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas
    from fdm.version import __version__
    sources = (
        "src/fdm/live_experiment/evidence_resources.py", "src/fdm/live_experiment/controller.py",
        "src/fdm/ui/live_measurement_canvas.py", "src/fdm/ui/trial_label_style.py",
        "scripts/validation/benchmark_trial_cleanup.py",
    )
    source_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    print("Loaded benchmark production modules: " + json.dumps(source_hashes), flush=True)

    if args.samples < 100 or args.files < 1000 or args.rounds < 1:
        raise ValueError("Use at least 100 samples, 1000 owned files, and one round (three for formal comparison).")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture_root = args.output.parent / ("owned-fixtures-" + str(time.time_ns()))
    fixture_root.mkdir()
    child_receipts = []
    cases = []
    for round_index in range(args.rounds):
        root = fixture_root / f"round-{round_index + 1}"
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--seed-child", str(root), "--files", str(args.files)],
            capture_output=True, text=True, check=True)
        receipt = json.loads(child.stdout.strip().splitlines()[-1])
        receipt["returncode"] = child.returncode
        child_receipts.append(receipt)
        # Alternate treatment order so a monotonically warming process does
        # not consistently favor either treatment.
        for enabled in ((False, True) if round_index % 2 == 0 else (True, False)):
            cases.append((round_index + 1, enabled, root))

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    clock = time.perf_counter
    pixels = np.random.default_rng(1825).integers(40, 220, size=(768, 1024, 3), dtype=np.uint8)
    fixture_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    report = {
        "schema": 1, "version": __version__, "hardware_validated": False,
        "formal_three_round_comparison": args.rounds >= 3,
        "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                        "qt_binding": qt_binding_version, "qt_platform": os.environ["QT_QPA_PLATFORM"],
                        "frame_size": [1024, 768], "frame_interval_ms": args.frame_ms,
                        "frame_sha256": fixture_digest, "samples": args.samples,
                        "warmup": args.warmup, "rounds": args.rounds, "files_per_round": args.files,
                        "source_sha256_at_import": source_hashes},
        "fixture_children": child_receipts, "cases": [],
        "boundaries": [
            "Controller and real canvas only; no sidebar, Excel, physical camera or monitor scan-out.",
            "Sequential free-line measurements run as fast as accepted; no harness sleep or processEvents in timing.",
            "Frames are delivered by a precise Qt timer; frame intervals measure GUI delivery, not camera sensor timing.",
            "Production collector batching and delays are unchanged; overlap is measured and reported explicitly.",
            "The initial child fixture preparation and window warmup are outside the measured interval.",
        ],
    }

    class TimedResources(EvidenceResources):
        def __init__(self, root):
            super().__init__(root)
            self.collect_started = None
            self.collect_finished = None
            self.first_delete = None
            self.last_delete = None
            self.collect_runs = []
            self.delete_times = []
        def _collect(self):
            start = clock()
            self.collect_started = start if self.collect_started is None else self.collect_started
            result = None
            try:
                result = super()._collect()
                return result
            finally:
                self.collect_finished = clock()
                self.collect_runs.append({"start": start, "end": self.collect_finished,
                                          "report": asdict(result) if result is not None else None})
        def _delete(self, *values, **kwargs):
            cleanup_report = values[3]
            count = cleanup_report.deleted_files
            super()._delete(*values, **kwargs)
            if cleanup_report.deleted_files > count:
                now = clock()
                self.first_delete = now if self.first_delete is None else self.first_delete
                self.last_delete = now
                self.delete_times.append(now)

    class Canvas(LiveMeasurementCanvas):
        def paintEvent(self, event):
            start = clock()
            super().paintEvent(event)
            end = clock()
            owner = getattr(self, "case", None)
            if owner is not None:
                owner.callback_samples("paint", start, end)
                owner.painted(end)

    class Case(QObject):
        def __init__(self, owner, round_index, enabled, root):
            super().__init__(owner)
            self.owner, self.round_index, self.enabled = owner, round_index, enabled
            self.resources = TimedResources(root)
            evidence_resources._resources = self.resources
            self.session = ExperimentSession(save_images=False, evidence_options={"mode": "disabled"})
            self.session.add_operator("Benchmark")
            self.session.set_materials(default_materials())
            self.controller = LiveExperimentController(self.session)
            self.controller.set_evidence_options({"mode": "disabled"})
            self.controller.set_calibration(CalibrationSnapshot(1., 1., source="simulated"))
            self.canvas = Canvas(self.controller)
            self.canvas.case = self
            self.canvas.resize(1100, 820)
            self.canvas.show()
            self.started = None
            self.finished = None
            self.pending = None
            self.accepted_total = 0
            self.submitted = 0
            self.sequence = 0
            self.rows = []
            self.frames = []
            self.callbacks = []
            self.paint_pending = []
            self.cleanup_future = None
            self.next_scheduled = False
            self.reported = False
            self.controller.recordAccepted.connect(owner.safe(self.accepted))
            self.controller.resultRejected.connect(lambda reason: owner.fail(RuntimeError(reason)))
            self.controller.measurementFeedback.connect(self.feedback)
            self.frame_timer = QTimer(self)
            self.frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.frame_timer.setInterval(args.frame_ms)
            self.frame_timer.timeout.connect(owner.safe(self.frame))
            self.completion_timer = QTimer(self)
            self.completion_timer.setInterval(10)
            self.completion_timer.timeout.connect(owner.safe(self.completed))
            self.frame_timer.start()
            self.completion_timer.start()
            QTimer.singleShot(0, owner.safe(self.frame))

        def feedback(self, item):
            # Real rejection is a failed comparison, not a silently dropped sample.
            if getattr(item, "kind", None) == "rejected":
                self.owner.fail(RuntimeError(str(item)))

        def callback_samples(self, kind, start, end):
            if self.started is not None and self.finished is None:
                self.callbacks.append({"kind": kind, "at": start, "ms": (end - start) * 1000})

        def frame(self):
            start = clock()
            self.sequence += 1
            frame = FrameHandle.snapshot(pixels, source_id="cleanup-benchmark", generation=1,
                                         sequence=self.sequence, live=True)
            if self.controller.receive_frame(frame):
                self.canvas.set_frame(frame)
            end = clock()
            if self.started is not None and self.finished is None:
                self.frames.append(start)
            self.callback_samples("frame_delivery", start, end)
            self.queue_next()

        def queue_next(self):
            if not self.next_scheduled and self.pending is None and self.finished is None:
                self.next_scheduled = True
                QTimer.singleShot(0, self.owner.safe(self.submit))

        def submit(self):
            self.next_scheduled = False
            if self.pending is not None or self.finished is not None:
                return
            if self.accepted_total >= args.warmup + args.samples:
                return
            if self.controller.state != "ready" or self.controller.presented is None:
                return
            if self.started is None and self.accepted_total >= args.warmup:
                self.started = clock()
                if self.enabled:
                    self.cleanup_future = self.resources.request_cleanup()
            index = self.submitted
            x, y = 65. + index % 18 * 48, 70. + (index // 18) % 12 * 45
            start = clock()
            self.controller.start((x, y))
            identity = self.controller.finish((x + 20., y + 2.), "wool", method="free")
            end = clock()
            self.pending = {"identity": identity, "submitted_at": start,
                            "number": self.submitted + 1, "submit_call_ms": (end - start) * 1000,
                            "measured": self.started is not None}
            self.submitted += 1
            self.callback_samples("measurement_submit", start, end)

        def accepted(self, record):
            start = clock()
            pending, self.pending = self.pending, None
            if pending is None:
                raise RuntimeError("Unexpected record acceptance without a submitted sample")
            self.accepted_total += 1
            if pending["measured"]:
                row = {**pending, "accepted_at": start,
                       "accept_ms": (start - pending["submitted_at"]) * 1000,
                       "diameter_um": record.diameter_um, "record_id": record.id}
                self.rows.append(row)
                self.paint_pending.append(row)
            self.canvas.update()
            self.queue_next()
            self.callback_samples("record_accepted", start, clock())

        def painted(self, now):
            for row in self.paint_pending:
                row["painted_at"] = now
                row["visible_ms"] = (now - row["submitted_at"]) * 1000
            self.paint_pending.clear()
            if self.accepted_total >= args.warmup + args.samples and self.finished is None:
                self.finished = now
            elif self.accepted_total == 0:
                self.queue_next()

        def completed(self):
            if self.finished is None or self.reported:
                return
            if self.cleanup_future is not None:
                current = self.resources._future
                if current is not None and not current.done():
                    return
                if current is not None:
                    current.result()  # A failed worker must not become an endless retry.
                # The production manager can finish one batch as deferred and
                # automatically resume in another future after an activity lease.
                # Never mistake that first partial response for completion.
                if sum(run["report"]["deleted_files"] for run in self.resources.collect_runs
                       if run["report"] is not None) < args.files:
                    self.cleanup_future = self.resources.request_cleanup()
                    return
            self.reported = True
            self.frame_timer.stop()
            self.completion_timer.stop()
            cleanup_runs = list(self.resources.collect_runs)
            cleanup = None
            if self.cleanup_future is not None:
                cleanup = {"deleted_files": 0, "deleted_bytes": 0, "in_use": 0, "skipped": 0,
                           "deferred": False, "errors": []}
                for run in cleanup_runs:
                    value = run["report"]
                    if value is None:
                        raise RuntimeError("Cleanup worker failed")
                    for key in ("deleted_files", "deleted_bytes", "in_use", "skipped"):
                        cleanup[key] += value[key]
                    cleanup["deferred"] |= value["deferred"]
                    cleanup["errors"].extend(value["errors"])
            if cleanup and cleanup["deleted_files"] != args.files:
                raise RuntimeError(f"Unexpected collection count: {cleanup}")
            begin, end = self.resources.collect_started, self.resources.collect_finished
            overlap = sum(max(0., min(self.finished, run["end"]) - max(self.started, run["start"]))
                          for run in cleanup_runs)
            deletes = self.resources.delete_times
            for row in self.rows:
                row["cleanup_overlap"] = any(min(row["painted_at"], run["end"]) >
                    max(row["submitted_at"], run["start"]) for run in cleanup_runs)
                row["unlink_overlap"] = (bisect_right(deletes, row["painted_at"]) >
                                         bisect_left(deletes, row["submitted_at"]))
            overlap_rows = [row for row in self.rows if row["cleanup_overlap"]]
            frame_intervals = [(b - a) * 1000 for a, b in zip(self.frames, self.frames[1:])]
            overlap_callbacks = [row["ms"] for row in self.callbacks if any(
                run["start"] <= row["at"] <= run["end"] for run in cleanup_runs)]
            result = {
                "round": self.round_index, "cleanup_enabled": self.enabled,
                "accepted_records": len(self.rows), "correct_values": all(
                    abs(row["diameter_um"] - (404 ** .5)) < 1e-9 for row in self.rows),
                "measurement_window_ms": (self.finished - self.started) * 1000,
                "frames": len(self.frames), "frame_interval": distribution(frame_intervals),
                "callback": distribution([row["ms"] for row in self.callbacks]),
                "callback_by_kind": {kind: distribution([row["ms"] for row in self.callbacks if row["kind"] == kind])
                                     for kind in {row["kind"] for row in self.callbacks}},
                "submit_call": distribution([row["submit_call_ms"] for row in self.rows]),
                "accept_latency": distribution([row["accept_ms"] for row in self.rows]),
                "visible_latency": distribution([row["visible_ms"] for row in self.rows]),
                "cleanup": cleanup, "cleanup_runs": cleanup_runs,
                "cleanup_worker_ms": sum(run["end"] - run["start"] for run in cleanup_runs) * 1000,
                "cleanup_span_ms": (end - begin) * 1000 if begin and end else 0.,
                "cleanup_overlap_ms": overlap * 1000,
                "cleanup_overlap_fraction": overlap / (self.finished - self.started),
                "cleanup_overlap_records": len(overlap_rows),
                "unlink_overlap_records": sum(row["unlink_overlap"] for row in self.rows),
                "deleted_during_measurements": bisect_right(deletes, self.finished) - bisect_left(deletes, self.started),
                "cleanup_overlap_callback": distribution(overlap_callbacks),
                "cleanup_overlap_accept_latency": distribution([row["accept_ms"] for row in overlap_rows]),
                "cleanup_overlap_visible_latency": distribution([row["visible_ms"] for row in overlap_rows]),
                "first_delete_at": self.resources.first_delete, "last_delete_at": self.resources.last_delete,
                "rows": self.rows, "frame_timestamps": self.frames, "callbacks": self.callbacks,
            }
            if len(self.rows) != args.samples or not result["correct_values"]:
                raise RuntimeError("Measurement completeness/value check failed")
            report["cases"].append(result)
            self.canvas.case = None
            self.canvas.close()
            self.controller.close()
            self.canvas.deleteLater()
            self.resources.shutdown()
            print(f"round {self.round_index} cleanup={self.enabled}: accept P95 "
                  f"{result['accept_latency']['p95_ms']:.3f} ms; overlap "
                  f"{len(overlap_rows)}/{len(self.rows)} records", flush=True)
            self.owner.persist()
            QTimer.singleShot(0, self.owner.safe(self.owner.next_case))

    class Runner(QObject):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.current = None
            self.failed = False
            self.watchdog = QTimer(self)
            self.watchdog.setSingleShot(True)
            self.watchdog.setInterval(120000)
            self.watchdog.timeout.connect(lambda: self.fail(RuntimeError("Benchmark case timed out")))
        def safe(self, callback):
            @wraps(callback)
            def run(*values):
                if self.failed:
                    return
                try:
                    return callback(*values)
                except BaseException as exc:
                    self.fail(exc)
            return run
        def persist(self):
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        def fail(self, exc):
            if self.failed:
                return
            self.failed = True
            report["error"] = {"message": str(exc), "traceback": traceback.format_exc()}
            self.persist()
            app.exit(1)
        def next_case(self):
            self.watchdog.start()
            if self.index < len(cases):
                self.current = Case(self, *cases[self.index])
                self.index += 1
                return
            self.watchdog.stop()
            report["aggregate"] = {}
            for enabled in (False, True):
                selected = [case for case in report["cases"] if case["cleanup_enabled"] == enabled]
                rows = [row for case in selected for row in case["rows"]]
                intervals = [(b - a) * 1000 for case in selected for a, b in
                             zip(case["frame_timestamps"], case["frame_timestamps"][1:])]
                report["aggregate"]["on" if enabled else "off"] = {
                    "accept_latency": distribution([row["accept_ms"] for row in rows]),
                    "visible_latency": distribution([row["visible_ms"] for row in rows]),
                    "frame_interval": distribution(intervals),
                    "callback": distribution([row["ms"] for case in selected for row in case["callbacks"]]),
                    "frame_delivery": distribution([row["ms"] for case in selected for row in case["callbacks"]
                                                     if row["kind"] == "frame_delivery"]),
                    "paint": distribution([row["ms"] for case in selected for row in case["callbacks"]
                                             if row["kind"] == "paint"]),
                    "cleanup_overlap_records": sum(case["cleanup_overlap_records"] for case in selected),
                }
            on, off = report["aggregate"]["on"], report["aggregate"]["off"]
            report["ratios_p95"] = {key: on[key]["p95_ms"] / off[key]["p95_ms"]
                for key in ("accept_latency", "visible_latency", "frame_interval", "callback", "frame_delivery", "paint")
                if on[key]["p95_ms"] is not None and off[key]["p95_ms"]}
            report["within_10_percent"] = {key: ratio <= 1.1 for key, ratio in report["ratios_p95"].items()}
            report["round_ratios_p95"] = []
            for index in range(1, args.rounds + 1):
                pair = {case["cleanup_enabled"]: case for case in report["cases"] if case["round"] == index}
                ratios = {key: pair[True][key]["p95_ms"] / pair[False][key]["p95_ms"]
                          for key in ("accept_latency", "visible_latency", "frame_interval", "callback")
                          if pair[True][key]["p95_ms"] is not None and pair[False][key]["p95_ms"]}
                report["round_ratios_p95"].append({"round": index, "ratios": ratios,
                    "within_10_percent": {key: value <= 1.1 for key, value in ratios.items()}})
            report["correctness_passed"] = True
            report["enough_overlap"] = all(case["cleanup_overlap_records"] >= 100
                for case in report["cases"] if case["cleanup_enabled"])
            self.persist()
            app.exit(0)

    runner = Runner()
    QTimer.singleShot(0, runner.safe(runner.next_case))
    code = app.exec()
    if runner.current is not None and not runner.current.reported:
        runner.current.frame_timer.stop()
        runner.current.controller.close()
        runner.current.canvas.close()
        runner.current.resources.shutdown()
    return code


if __name__ == "__main__":
    options = parse_args()
    if options.seed_child is not None:
        seed_child(options.seed_child, options.files)
    raise SystemExit(main(options))
