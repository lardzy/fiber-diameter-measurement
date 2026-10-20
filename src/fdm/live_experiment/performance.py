"""Bounded scalar diagnostics; no images, capture timestamps or project edits."""
from __future__ import annotations

from collections import deque
import math


METRICS = {
    "analysis_ms": ("ms", "Controller submission to result consumption; includes scheduling, process startup, shared memory, service, IPC and Qt polling."),
    "alignment_ms": ("ms", "Independent fixed-reference alignment function wall time; excludes executor queue wait."),
    "candidate_verification_ready_ms": ("ms", "Alignment submission to parent worker result readiness; includes serial queue, process/IPC and algorithm, excludes Qt consumption."),
    "candidate_verification_gui_wait_ms": ("ms", "Parent result readiness to GUI consumption, separately measured from verification work."),
    "candidate_verification_delivery_ms": ("ms", "Alignment submission to GUI consumption, including executor and Qt scheduling; measured for the actual checked frame."),
    "candidate_available_ms": ("ms", "Completed continuous candidate presentation interval before a latched pause; excludes unfinished intervals."),
    "candidate_unavailable_ms": ("ms", "Completed latched candidate pause until sustained timely recovery; live image presentation continues."),
    "click_to_accept_ms": ("ms", "Candidate intent creation to ledger insertion; includes queued clicks and motion/alignment verification, excludes subsequent UI painting."),
    "stable_wait_ms": ("ms", "Additional automatic-analysis stable interval after global readiness and first successful independent alignment; excludes earlier field settling."),
    "analysis_to_paint_ms": ("ms", "Controller submission to first paint containing candidates; includes analysis and presentation delay, not isolated paint CPU time."),
    "ingest_fps": ("frames/s", "Reciprocal intervals between distinct live frames admitted by the controller, using parent perf_counter; upstream mailbox drops are not counted."),
    "presented_fps": ("frames/s", "Reciprocal intervals between distinct live frames completing Qt paint; repeated paints of one frame are excluded."),
    "frame_age_ms": ("ms", "First completed Qt paint minus FrameHandle.received_at on the OS performance-counter timeline. Microview uses SDK callback buffer-copy completion (QPC); other adapters use application receipt. NOT sensor exposure age."),
    "analysis_service_total_ms": ("ms", "Child analyze_frame total wall time, including input preparation and native analysis."),
    "analysis_preparation_ms": ("ms", "Child analyze_frame input validation/native loading/pixel preparation before native invocation."),
    "analysis_processing_ms": ("ms", "Service total minus preparation; native invocation and result wrapping, not an isolated native kernel benchmark."),
    "analysis_shared_memory_ms": ("ms", "Parent shared-memory allocation and source-pixel copy."),
    "analysis_budget_wait_ms": ("ms", "Parent wait for the shared heavy-compute permit after pixel copy; excludes process startup."),
    "analysis_process_start_call_ms": ("ms", "Cold starts only: parent Pipe/Event/process.start call. Does not include all child imports or readiness."),
    "analysis_dispatch_to_receive_ms": ("ms", "Parent command-send start to reader recv completion/deserialization; includes child startup/imports, service, IPC and scheduling."),
    "analysis_startup_transport_overhead_ms": ("ms", "Dispatch-to-receive minus child service total: combined startup/imports, shared-memory attachment, IPC/serialization and scheduling; these components are NOT independently separated. Unavailable for a missing service duration or negative duration difference, never clamped to zero."),
}


class LivePerformanceDiagnostics:
    LIMIT = 512

    def __init__(self):
        self.samples = {name: deque(maxlen=self.LIMIT) for name in METRICS}
        self.streams = {name: deque(maxlen=self.LIMIT) for name in ("ingest", "presented")}
        self.source = None
        self.live = False
        self.stopped = True
        self.invalid_frame_ages = 0

    def record(self, name, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        if math.isfinite(value) and value >= 0:
            self.samples[name].append(float(value))

    def received(self, frame, now):
        source = (frame.source_id, frame.generation, frame.size)
        if source != self.source or frame.live != self.live or self.stopped:
            self.source, self.live = source, frame.live
            for stream in self.streams.values():
                stream.clear()
            for metric in ("ingest_fps", "presented_fps", "frame_age_ms"):
                self.samples[metric].clear()
            self.invalid_frame_ages = 0
        self.stopped = False
        if frame.live:
            self._frame("ingest", frame.sequence, now)

    def presented(self, frame, now):
        if (not frame.live or self.stopped or
                (frame.source_id, frame.generation, frame.size) != self.source):
            return
        if self._frame("presented", frame.sequence, now):
            age = (now - frame.received_at) * 1000
            if math.isfinite(age) and age >= 0:
                self.record("frame_age_ms", age)
            else:
                self.invalid_frame_ages += 1

    def _frame(self, name, sequence, now):
        stream = self.streams[name]
        if stream and sequence <= stream[-1][0]:
            return False
        if stream and now > stream[-1][1]:
            self.record(name + "_fps", 1 / (now - stream[-1][1]))
        stream.append((sequence, now))
        return True

    def stop(self):
        # Retain the last run for offline export after the user stops capture.
        self.stopped = True

    def analysis(self, response):
        service = (response.get("result") or {}).get("diagnostics") or {}
        for key, field in (("analysis_service_total_ms", "total_ms"),
                           ("analysis_preparation_ms", "preparation_ms")):
            self.record(key, service.get(field))
        total, prepare = service.get("total_ms"), service.get("preparation_ms")
        if (isinstance(total, (int, float)) and isinstance(prepare, (int, float))):
            self.record("analysis_processing_ms", total - prepare)
        timings = response.get("timings") or {}
        for name in ("shared_memory_ms", "budget_wait_ms", "process_start_call_ms",
                     "dispatch_to_receive_ms", "startup_transport_overhead_ms"):
            self.record("analysis_" + name, timings.get(name))

    def summary(self):
        result = {}
        for name, history in self.samples.items():
            values = sorted(history)
            def percentile(fraction):
                if not values:
                    return None
                index = (len(values) - 1) * fraction
                lower = int(index)
                upper = min(lower + 1, len(values) - 1)
                return values[lower] + (values[upper] - values[lower]) * (index - lower)
            result[name] = {"n": len(values), "p50": percentile(.5), "p95": percentile(.95)}
        return result

    @classmethod
    def schema(cls):
        return {"version": 1, "sample_limit_per_metric": cls.LIMIT,
                "percentiles": "Linear interpolation over retained samples; n=0 yields null, not zero.",
                "clock": "Qt/controller use parent perf_counter. FrameHandle receipt uses the same OS counter (Microview native QPC). Worker timings use same-process durations only, with no cross-process timestamp subtraction.",
                "frame_scope": "Most recent admitted source/generation and capture run; retained after stop, reset at next run. Analysis metrics span this workspace lifetime.",
                "fps": "P50/P95 use instantaneous inverse intervals; frame_sampling.window_fps uses (retained_frames-1)/elapsed_span. Neither is camera sensor FPS.",
                "limitations": ["No sensor exposure timestamp", "No isolated Qt paint CPU measurement",
                    "Child startup and IPC overhead are combined, not individually measured",
                    "Synthetic and real-device runs require separate baseline comparison"],
                "metrics": {name: {"unit": unit, "definition": definition}
                            for name, (unit, definition) in METRICS.items()}}

    def frame_context(self):
        result = {"source": None if self.source is None else {
            "source_id": self.source[0], "generation": self.source[1], "size": list(self.source[2])},
            "live_source": self.live, "stopped": self.stopped,
            "invalid_frame_age_samples": self.invalid_frame_ages}
        for name, stream in self.streams.items():
            span = stream[-1][1] - stream[0][1] if len(stream) > 1 else 0.
            result[name] = {"frames": len(stream), "span_seconds": span,
                "first_sequence": stream[0][0] if stream else None,
                "last_sequence": stream[-1][0] if stream else None,
                "window_fps": (len(stream) - 1) / span if span > 0 else None}
        return result
