"""Serial, reproducible CU-port benchmarks and a live-controller endurance probe.

Examples (use the installed native build on the target device)::

    python scripts/benchmark_cu_live_measurement.py --output results/cu-live.json
    python scripts/benchmark_cu_live_measurement.py --modes soak \
        --duration-seconds 3600 --output results/cu-live-hour.json

Default numerical runs warm up five times and collect thirty samples. Run this
script alone, after correctness tests. Synthetic fixtures establish runtime
costs, not real-camera metrology, display latency or installed-Windows acceptance.
The soak drives the actual controller, motion detector, assisted native kernel,
ledger and PNG evidence writer, but supplies synthetic frames without painting a
QWidget. Its heartbeat measures event-loop responsiveness, not screen refresh.
Install the optional benchmark dependency (``uv pip install psutil`` or the
project's ``benchmark`` extra) for current and sampled peak parent/worker RSS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time

# Set before importing NumPy/OpenCV; identical defaults in the spawned worker.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {"count": len(array), "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(array.max()), "samples_ms": array.tolist()}


class BoundedSamples:
    """Deterministic reservoir prevents an hour's metrics dominating RSS."""
    def __init__(self, limit=100000):
        self.values = []
        self.count = 0
        self.maximum = 0.0
        self.limit = limit
        self.random = random.Random(20261015)

    def append(self, value):
        self.count += 1
        self.maximum = max(self.maximum, value)
        if len(self.values) < self.limit:
            self.values.append(value)
        else:
            index = self.random.randrange(self.count)
            if index < self.limit:
                self.values[index] = value

    def report(self):
        result = distribution(self.values)
        result.pop("samples_ms", None)
        result.update(observations=self.count, exact_max_ms=self.maximum,
                      sampling="deterministic reservoir; percentiles use at most 100000 events")
        return result


def resource_snapshot():
    """RSS is sampled; platform peak includes earlier cases and is labelled so."""
    result = {"process_cpu_seconds": time.process_time(), "python_threads": threading.active_count()}
    try:
        import psutil
        process = psutil.Process()
        result.update(rss_bytes=process.memory_info().rss, native_threads=process.num_threads())
        children = []
        for child in process.children(recursive=True):
            try:
                with child.oneshot():
                    children.append({"pid": child.pid, "name": child.name(),
                        "rss_bytes": child.memory_info().rss,
                        "cpu_seconds": sum(child.cpu_times()[:2]),
                        "native_threads": child.num_threads()})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Automatic workers may exit between process enumeration and
                # sampling. Preserve valid parent/other worker measurements.
                continue
        result["children"] = children
        result["children_rss_bytes"] = sum(child["rss_bytes"] for child in children)
        result["total_rss_bytes"] = result["rss_bytes"] + result["children_rss_bytes"]
        result["children_cpu_seconds"] = sum(child["cpu_seconds"] for child in children)
        result["resource_provider"] = "psutil"
    except (ImportError, OSError):
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
            if memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                result.update(rss_bytes=counters.WorkingSetSize, process_lifetime_peak_rss_bytes=counters.PeakWorkingSetSize)
            result["resource_provider"] = "Win32; child RSS unavailable without psutil"
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["process_lifetime_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
            result["resource_provider"] = "getrusage; current/child RSS unavailable without psutil"
    return result


class ResourceSampler:
    def __init__(self, *, trace=False):
        self.maxima = {}
        self.children = {}
        self.sample_count = 0
        self.trace = trace
        self.sample_spans = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="cu-benchmark-resources", daemon=True)

    def _run(self):
        while not self.stop.wait(.1):
            started = time.perf_counter()
            try:
                sample = resource_snapshot()
                self._record(sample)
            except Exception:
                # A worker may exit during process enumeration; next sample is valid.
                pass
            finally:
                if self.trace:
                    self.sample_spans.append({"started_at": started, "finished_at": time.perf_counter()})

    def _record(self, sample):
        self.sample_count += 1
        for key in ("rss_bytes", "children_rss_bytes", "total_rss_bytes", "native_threads", "process_lifetime_peak_rss_bytes"):
            if key in sample:
                self.maxima[key] = max(self.maxima.get(key, 0), sample[key])
        for child in sample.get("children", []):
            peak = self.children.setdefault(child["pid"], {"pid": child["pid"], "name": child["name"],
                "sampled_peak_rss_bytes": 0, "maximum_observed_cpu_seconds": 0, "sampled_peak_native_threads": 0})
            peak["sampled_peak_rss_bytes"] = max(peak["sampled_peak_rss_bytes"], child["rss_bytes"])
            peak["maximum_observed_cpu_seconds"] = max(peak["maximum_observed_cpu_seconds"], child["cpu_seconds"])
            peak["sampled_peak_native_threads"] = max(peak["sampled_peak_native_threads"], child["native_threads"])

    def __enter__(self):
        self.started = time.perf_counter()
        self.before = resource_snapshot()
        self._record(self.before)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        after = resource_snapshot()
        self._record(after)
        samples = [self.before, after, self.maxima]
        self.result = {"wall_seconds": time.perf_counter() - self.started,
            "process_cpu_seconds": after["process_cpu_seconds"] - self.before["process_cpu_seconds"],
            "before": self.before, "after": after, "sampling_interval_ms": 100,
            "sample_count": self.sample_count, "children_sampled_peaks": list(self.children.values()),
            "peak_scope": "Sampled current RSS within this case; sub-100 ms peaks may be missed. Lifetime peaks are separately named."}
        for key in ("rss_bytes", "children_rss_bytes", "total_rss_bytes", "native_threads", "process_lifetime_peak_rss_bytes"):
            values = [sample[key] for sample in samples if key in sample]
            self.result["sampled_max_" + key] = max(values) if values else None
        if self.trace:
            self.result["sampling_spans"] = self.sample_spans


def fixture(size, *, kind="strip", width=20, height=None):
    height = height or max(240, size * 2 // 3)
    image = np.full((height, size, 3), 240, np.uint8)
    center = height // 2
    if kind == "sparse":
        for y in np.linspace(height // 8, height * 7 // 8, 8).astype(int):
            image[y - width // 2:y + width // 2, size // 8:size * 7 // 8] = 30
    else:
        image[center - width // 2:center + width // 2, size // 8:size * 7 // 8] = 30
        if kind == "cross":
            image[height // 8:height * 7 // 8, size // 2 - width // 2:size // 2 + width // 2] = 30
    image.setflags(write=False)
    return image, (size // 2, center - 50), (size // 2, center + 50)


def input_digest(image):
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode())
    digest.update(image.dtype.str.encode())
    digest.update(memoryview(image).cast("B"))
    return digest.hexdigest()


def timed(call, warmup, samples):
    for _ in range(warmup):
        call()
    times, results = [], []
    with ResourceSampler() as resources:
        for _ in range(samples):
            started = time.perf_counter()
            result = call()
            times.append((time.perf_counter() - started) * 1000)
            results.append(result)
    return distribution(times), results, resources.result


def manual_bench(args):
    from fdm.live_experiment.frames import FrameHandle
    from fdm.services.cu_manual_measurement import measure_manual
    rows = []
    for size in args.sizes:
        pixels, start, end = fixture(size)
        def snapshot():
            return FrameHandle.snapshot(pixels, source_id="benchmark", live=False)
        def snapshot_only():
            snapshot()  # Do not retain thirty 6K snapshots in the benchmark.
        def complete():
            owned = snapshot()
            return measure_manual(owned.pixels, start, end)
        for name, call in (("snapshot_copy", snapshot_only),
                           ("native_service", lambda: measure_manual(pixels, start, end)),
                           ("snapshot_plus_native_service", complete)):
            metrics, results, resources = timed(call, args.warmup, args.samples)
            measured = [result for result in results if isinstance(result, dict)]
            rows.append({"size": size, "shape": pixels.shape, "input_sha256": input_digest(pixels),
                "stage": name, "timing": metrics, "resources": resources,
                "statuses": dict(Counter(result["status"] for result in measured)),
                "widths_px": sorted({result["width_px"] for result in measured if result.get("width_px") is not None}),
                "diagnostics": measured[-1].get("diagnostics") if measured else None})
            if measured and any(result["status"] != "ok" or result["width_px"] != 19 for result in measured):
                raise AssertionError("Known-width manual fixture changed; timing cannot be accepted")
    return rows


def statistics_bench(args):
    from fdm.live_experiment import ExperimentSession, default_materials, summarize
    rows = []
    for count in (1, 100, 1000, 10000):
        session = ExperimentSession(save_images=False)
        session.add_operator("benchmark")
        session.set_materials(default_materials())
        for index in range(count):
            session.add_measurement(10 + (index % 100) / 10, "wool")
        summarize(session)
        anchor = next(iter(session.measurements))
        def append_and_summary():
            record = session.add_measurement(17.25, "wool")
            result = summarize(session)
            session.undo()
            return result
        def edit_and_summary():
            session.edit_measurement(anchor, diameter_um=15.25)
            result = summarize(session)
            session.undo()
            return result
        for name, call in (("cached_summary", lambda: summarize(session)),
                           ("append_summary_undo", append_and_summary),
                           ("edit_summary_undo", edit_and_summary)):
            metrics, results, resources = timed(call, args.warmup, args.samples)
            rows.append({"history_count": count, "stage": name, "timing": metrics,
                         "status": results[-1]["status"], "resources": resources})
        assert len(session.measurements) == count
    return rows


def automatic_bench(args):
    from fdm.services.cu_automatic_worker import CuAutomaticWorker
    rows = []
    cases = [("single8", 320, "strip", 8), ("single20", 320, "strip", 20),
             ("cross", 640, "cross", 12), ("large_sparse8", args.automatic_size, "sparse", 8)]
    worker = CuAutomaticWorker()
    try:
        for name, size, kind, width in cases:
            pixels, _, _ = fixture(size, kind=kind, width=width)
            for algorithm in ("traditional", "multiscale"):
                def analyze():
                    task_id = worker.submit(pixels, algorithm=algorithm, memory_budget=args.automatic_budget_mib * 1024**2)
                    until = time.monotonic() + 35
                    while time.monotonic() < until:
                        for response in worker.poll():
                            if response["task_id"] == task_id:
                                return response
                        time.sleep(.001)
                    worker.cancel(task_id)
                    raise RuntimeError("Automatic watchdog failed to respond within 35 seconds")
                metrics, responses, resources = timed(analyze, args.warmup, args.samples)
                results = [response["result"] for response in responses]
                native_ms = [result["diagnostics"]["total_ms"] for result in results
                             if "total_ms" in result.get("diagnostics", {})]
                rows.append({"case": name, "shape": pixels.shape, "input_sha256": input_digest(pixels),
                    "algorithm": algorithm, "worker_roundtrip": metrics,
                    "service_inside_worker": distribution(native_ms), "resources": resources,
                    "statuses": dict(Counter(result["status"] for result in results)),
                    "candidate_counts": [len(result.get("candidates", [])) for result in results],
                    "last_diagnostics": results[-1].get("diagnostics"),
                    "errors": [response["error"] for response in responses if response.get("error")]})
    finally:
        worker.close()
    return rows


def soak_bench(args):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fdm.live_experiment import ExperimentSession, default_materials
    from fdm.live_experiment.controller import CalibrationSnapshot, LiveExperimentController
    from fdm.live_experiment.frames import FrameHandle, PresentedFrameToken
    application = QApplication.instance() or QApplication([])
    from fdm.services.cu_automatic_worker import CuAutomaticWorker
    automatic = CuAutomaticWorker() if args.soak_automatic else None
    automatic_results = Counter()
    automatic_latency = BoundedSamples()
    automatic_started = None
    automatic_due = time.perf_counter()
    pixels, start, end = fixture(args.soak_size)
    moved = np.roll(pixels, 12, axis=1)
    moved.setflags(write=False)
    session = ExperimentSession(save_images=True)
    session.add_operator("benchmark")
    session.set_materials(default_materials())
    errors = Counter()
    accepted_latency, handoff, heartbeat = (BoundedSamples() for _ in range(3))
    quarter_latency = [BoundedSamples() for _ in range(4)]
    total_frames = attempted = 0
    output_directory = Path(args.evidence_dir) if args.evidence_dir else None
    with tempfile.TemporaryDirectory(prefix="fdm-cu-soak-") as temporary:
        directory = output_directory or Path(temporary)
        controller = LiveExperimentController(session, evidence_directory=directory)
        controller.set_calibration(CalibrationSnapshot(1, 1, 1, "synthetic benchmark"))
        controller.resultRejected.connect(lambda value: errors.update([str(value)]))
        started = last_tick = time.perf_counter()
        def record_accepted(record):
            latency = record.provenance["latency_ms"]
            accepted_latency.append(latency)
            quarter = min(3, int((time.perf_counter() - started) / args.duration_seconds * 4))
            quarter_latency[quarter].append(latency)
        controller.recordAccepted.connect(record_accepted)
        frame_due = measure_due = started
        progress_due = started + 60
        transitions = [0]
        controller.fieldChanged.connect(lambda *_: transitions.__setitem__(0, transitions[0] + 1))
        try:
            with ResourceSampler() as resources:
                while time.perf_counter() - started < args.duration_seconds:
                    now = time.perf_counter()
                    if now >= progress_due:
                        print(f"Soak {now - started:.0f}s: {len(session.measurements)} accepted, "
                              f"{total_frames} frames, {sum(errors.values())} rejected/errors", flush=True)
                        progress_due = now + 60
                    heartbeat.append((now - last_tick) * 1000)
                    last_tick = now
                    if now >= frame_due:
                        before = time.perf_counter()
                        # Translation every ten seconds exercises move invalidation.
                        source = moved if int((now - started) // 10) % 2 else pixels
                        frame = FrameHandle.snapshot(source, source_id="synthetic-soak", sequence=total_frames)
                        controller.receive_frame(frame)
                        controller.mark_presented(PresentedFrameToken(frame, controller.field_epoch, total_frames, 1, (0, 0)))
                        handoff.append((time.perf_counter() - before) * 1000)
                        total_frames += 1
                        frame_due = now + 1 / args.frame_hz  # latest-only, no catch-up queue
                    if now >= measure_due:
                        if controller.state == "ready" and len(controller._jobs) < 2:
                            attempted += 1
                            try:
                                controller.start(start)
                                controller.finish(end, "wool")
                            except (RuntimeError, ValueError) as exc:
                                errors.update([str(exc)])
                        measure_due = now + 1 / args.measurement_hz
                    application.processEvents()
                    controller.poll()
                    if automatic is not None:
                        for response in automatic.poll():
                            automatic_results[(response.get("result") or {}).get("status", "error")] += 1
                            if automatic_started is not None:
                                automatic_latency.append((time.perf_counter() - automatic_started) * 1000)
                            automatic_started = None
                        if automatic_started is None and now >= automatic_due:
                            automatic.submit(pixels, "traditional")
                            automatic_started = time.perf_counter()
                            automatic_due = now + 10
                    time.sleep(.001)
                # Stop new intents, allow accepted computations/evidence to finish.
                drain_until = time.perf_counter() + 30
                while (controller._jobs or controller.evidence.pending) and time.perf_counter() < drain_until:
                    application.processEvents()
                    controller.poll()
                    time.sleep(.002)
                unfinished = bool(controller._jobs or controller.evidence.pending)
            evidence_states = Counter((record.evidence or {}).get("state", "missing") for record in session.measurements.values())
            files = list(directory.glob("*.png"))
            report = {"requested_seconds": args.duration_seconds, "elapsed_seconds": time.perf_counter() - started,
                "shape": pixels.shape, "input_sha256": input_digest(pixels), "frame_hz_requested": args.frame_hz,
                "measurement_hz_requested": args.measurement_hz, "frames_delivered": total_frames,
                "measurement_attempts": attempted, "accepted_records": len(session.measurements),
                "field_transitions": transitions[0], "unfinished_at_drain_deadline": unfinished,
                "frame_snapshot_and_handoff": handoff.report(), "event_loop_gap": heartbeat.report(),
                "manual_submit_to_accept": accepted_latency.report(), "evidence_states": dict(evidence_states),
                "latency_by_elapsed_quarter": [sample.report() for sample in quarter_latency],
                "accepted_event_count_matches_ledger": accepted_latency.count == len(session.measurements),
                "all_records_have_valid_field": all(record.field_id in session.fields for record in session.measurements.values()),
                "evidence_files": len(files), "evidence_bytes": sum(path.stat().st_size for path in files),
                "evidence_retained_at": str(directory.resolve()) if output_directory else None,
                "frame_budget_peak_bytes": controller.budget.peak_bytes, "resources": resources.result,
                "background_automatic_enabled": automatic is not None,
                "background_automatic_results": dict(automatic_results),
                "background_automatic_latency": automatic_latency.report(),
                "errors": dict(errors), "boundary": "synthetic headless controller; no real camera, QPainter or installed-package acceptance"}
        finally:
            controller.close()
            if automatic is not None:
                automatic.close()
    return report


def environment():
    from fdm import _native
    native = _native.require_native(validate_source=True)
    paths = [*sorted((ROOT / "native").glob("**/*.*")),
             *sorted((ROOT / "src/fdm/live_experiment").glob("*.py")),
             ROOT / "src/fdm/services/cu_manual_measurement.py", ROOT / "src/fdm/services/cu_automatic_measurement.py",
             ROOT / "src/fdm/services/cu_automatic_worker.py"]
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".hpp")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    dependencies = {}
    for name in ("numpy", "opencv-python", "PySide6", "pybind11", "psutil", "zlib-ng"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "python": sys.version, "executable": sys.executable, "cpu_count": os.cpu_count(),
        "dependencies": dependencies, "opencv_runtime": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "git_head": commit, "source_sha256": sources, "native_build_info": native.build_info()}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("manual", "statistics", "automatic", "soak"), default=["manual", "statistics", "automatic"])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--automatic-size", type=int, default=2048)
    parser.add_argument("--automatic-budget-mib", type=int, default=2048)
    parser.add_argument("--duration-seconds", type=float, default=3600)
    parser.add_argument("--soak-size", type=int, default=2048)
    parser.add_argument("--frame-hz", type=float, default=30)
    parser.add_argument("--measurement-hz", type=float, default=2)
    parser.add_argument("--soak-automatic", action="store_true", help="Run a full-image background task every ten seconds during the soak")
    parser.add_argument("--evidence-dir", type=Path, help="Retain soak PNGs in this directory; default removes temporary PNGs after reporting")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 0 or min(args.sizes + [args.automatic_size, args.soak_size]) < 320:
        parser.error("samples>=1, warmup>=0 and image widths>=320 are required")
    if min(args.duration_seconds, args.frame_hz, args.measurement_hz, args.automatic_budget_mib) <= 0:
        parser.error("duration, rates and automatic budget must be positive")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    cv2.setNumThreads(1)
    report = {"schema_version": 1, "environment": environment(),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "validation_boundary": "Synthetic performance only; correctness, real-camera and installed-Windows acceptance are separate."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    methods = {"manual": manual_bench, "statistics": statistics_bench, "automatic": automatic_bench, "soak": soak_bench}
    for mode in dict.fromkeys(args.modes):
        print(f"Running {mode} serially...", flush=True)
        report[mode] = methods[mode](args)
        # Persist each completed mode so an interrupted hour probe retains earlier results.
        temporary = args.output.with_suffix(args.output.suffix + ".pending")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
        os.replace(temporary, args.output)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
