"""Trace candidate visibility under controlled alignment scheduling pressure.

Run each case in a fresh process, for example::

    QT_QPA_PLATFORM=offscreen .venv/bin/python \
      scripts/validation/validate_candidate_presentation.py \
      --delay-ms 80 --output /tmp/candidate-presentation-80.json

This uses actual Qt painting, frame ownership, alignment and JPEG writes, with
identical 768x576 source pixels at 30 Hz and a fixed candidate fixture. The
injected delay represents scheduling pressure, not a real menu or camera test.
It changes no user preferences, projects or production algorithm thresholds.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-ms", type=int, choices=(0, 40, 80, 200), default=0)
    parser.add_argument("--mode", choices=("baseline", "guard"), default="guard")
    parser.add_argument("--duration", type=float, default=6.)
    parser.add_argument("--recovery-seconds", type=float, default=0.,
                        help="Continue preview after stopping new measurements; original trace acceptance remains unchanged.")
    parser.add_argument("--measure-ms", type=int, default=150,
                        help="Continuous accepted manual measurements and JPEG writes (default: 150 ms).")
    parser.add_argument("--gallery", action="store_true", help="Keep the actual evidence gallery open while saving.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration < 5 or args.measure_ms < 20 or not 0 <= args.recovery_seconds <= 60:
        parser.error("duration must be at least 5 seconds, measure-ms at least 20, recovery-seconds in [0,60]")

    import fdm.live_experiment.candidate_alignment as alignment
    from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    work = alignment.CandidateAlignmentMonitor._work
    original_paint = LiveMeasurementCanvas.paintEvent
    paints = []
    delay_injections = []
    batch_ids = set()

    def delayed(monitor, *values, **kwargs):
        # Inject at the parent monitor boundary: a spawned worker intentionally
        # does not inherit monkeypatches of compare_candidate_frames. _run_work
        # stamps ready_at only after this hook returns, so delay remains inside
        # the genuine submit-to-ready budget.
        result = work(monitor, *values, **kwargs)
        if args.delay_ms and 1.5 <= time.perf_counter() - started < 3.:
            wait_started = time.perf_counter()
            time.sleep(args.delay_ms / 1000.)
            waited_ms = (time.perf_counter() - wait_started) * 1000
            delay_injections.append({"started_seconds": wait_started - started, "elapsed_ms": waited_ms})
            return replace(result, elapsed_ms=result.elapsed_ms + waited_ms)
        return result

    def traced(canvas, event):
        original_paint(canvas, event)
        if canvas.frame is not None:
            paints.append({"seconds": time.perf_counter() - started,
                           "sequence": canvas.frame.sequence,
                           "visible": bool(canvas._candidates_visible())})
        context = canvas.controller._automatic_context
        if context and context.get("batch_id"):
            batch_ids.add(context["batch_id"])

    alignment.CandidateAlignmentMonitor._work = delayed
    LiveMeasurementCanvas.paintEvent = traced
    path = Path(__file__).with_name("benchmark_candidate_preview.py")
    spec = importlib.util.spec_from_file_location("candidate_presentation_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    sys.argv = [str(path), "--mode", args.mode, "--duration", str(args.duration),
                "--sample-ms", "0", "--measure-ms", str(args.measure_ms),
                "--recovery-seconds", str(args.recovery_seconds), "--output", str(args.output)]
    if args.gallery:
        sys.argv.append("--gallery")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = benchmark.main()
    finally:
        alignment.CandidateAlignmentMonitor._work = work
        LiveMeasurementCanvas.paintEvent = original_paint
    report = json.loads(args.output.read_text(encoding="utf-8"))
    transitions = [current for previous, current in zip(paints, paints[1:])
                   if previous["visible"] != current["visible"]]
    after_recovery = [entry for entry in paints if entry["seconds"] >= 3.8]
    trace_ok = (args.mode == "baseline" or
                (len(transitions) <= (3 if args.delay_ms else 1)
                 and bool(after_recovery) and all(entry["visible"] for entry in after_recovery)
                 and len(batch_ids) == 1 and (not args.delay_ms or bool(delay_injections))))
    report["presentation_trace"] = {
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed pixels and candidate fixture. Real Qt/alignment/acceptance/JPEG. No real camera or native-menu acceptance.",
        "injected_alignment_delay_ms": args.delay_ms,
        "injection_interval_seconds": [1.5, 3.],
        "actual_delay_injections": delay_injections,
        "visibility_transitions": transitions,
        "visibility_transition_count": len(transitions),
        "batch_count": len(batch_ids),
        "post_load_recovery_seconds": args.recovery_seconds,
        "recovered_continuously_visible": bool(after_recovery) and all(entry["visible"] for entry in after_recovery),
        "paints": paints,
        "passed": trace_ok and code == 0,
    }
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "src/fdm/live_experiment/controller.py", "src/fdm/live_experiment/candidate_alignment.py",
        "src/fdm/live_experiment/candidate_alignment_worker.py",
        "src/fdm/live_experiment/performance.py", "src/fdm/ui/live_measurement_canvas.py",
        "src/fdm/ui/measurement_evidence_gallery.py",
        "scripts/validation/benchmark_candidate_preview.py",
        "scripts/validation/validate_candidate_presentation.py")}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode": args.mode, "delay_ms": args.delay_ms,
                      "passed": report["presentation_trace"]["passed"], "fps": report["fps"],
                      "p95_frame_age_ms": report["p95_frame_age_ms"], "transitions": len(transitions),
                      "remaining_budget_bytes": report["remaining_budget_bytes"]}, ensure_ascii=False))
    return int(not report["presentation_trace"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
