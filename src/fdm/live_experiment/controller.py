"""GUI-owned field state with isolated latest-only motion and FIFO measurement.

Only poll() publishes results. A completed calculation is not an accepted
record until its source, field, calibration and latest presented frame pass.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import threading
import time
import uuid

from PySide6.QtCore import QObject, QTimer, Signal, Slot, Qt

from .frames import FrameHandle, FrameBudget, FrameBudgetExceeded, PresentedFrameToken, acquisition_reservation
from .motion import FieldMotionDetector, MotionPreparer, MOTION_REVISION, reduced_gray
from .evidence import EvidenceStore
from .feedback import feedback_for_reason
from .candidate_alignment import CandidateAlignmentMonitor
from .evidence_options import measurement_evidence_points
from .performance import LivePerformanceDiagnostics


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    x_um: float
    y_um: float
    revision: int = 0
    source: str = ""

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.x_um, self.y_um)):
            raise ValueError("标定必须是有限正数。")

    @property
    def isotropic(self):
        return math.isclose(self.x_um, self.y_um, rel_tol=1e-12)

    @classmethod
    def from_calibration(cls, calibration, revision=0):
        from fdm.units import millimeters_per_unit
        factor = millimeters_per_unit(calibration.unit)
        if factor is None:
            raise ValueError("含量试验需要可换算为微米的标定单位。")
        return cls(calibration.pixel_size_x * factor * 1000,
                   calibration.pixel_size_y * factor * 1000,
                   revision, calibration.source_label)


@dataclass(slots=True)
class MeasurementIntent:
    id: str
    token: PresentedFrameToken
    start: tuple[float, float]
    end: tuple[float, float]
    calibration: CalibrationSnapshot
    operator_id: str
    direction: str
    material_id: str
    method: str
    cancel: threading.Event
    lease: str
    submitted_at: float
    submission_barrier_sequence: int | None = None
    submission_barrier_at: float | None = None
    range_confirmation: dict | None = None
    submission_barrier_initial_sequence: int | None = None
    submission_barrier_previous_sequence: int | None = None
    measurement_number: int | None = None
    classification_generation: int = 0


@dataclass(slots=True)
class _MotionFrame:
    frame: FrameHandle | None
    lease: str | None
    prepared: object = None
    lock: object = field(default_factory=threading.Lock)
    armed: bool = False


class LiveExperimentController(QObject):
    changed = Signal()
    fieldChanged = Signal(int, str)
    statusChanged = Signal(str)
    recordAccepted = Signal(object)
    recordsAccepted = Signal(object)
    resultRejected = Signal(str)
    measurementFeedback = Signal(object)
    feedbackInvalidated = Signal(str)
    candidatesChanged = Signal(object)
    candidateStateChanged = Signal()
    frameReady = Signal(object)
    _completionAvailable = Signal()

    def __init__(self, session, *, evidence_directory=None, parent=None, manual_service=None,
                 verify_before_present=False):
        super().__init__(parent)
        self.session = session
        self.budget = FrameBudget()
        self.evidence = EvidenceStore(evidence_directory, budget=self.budget)
        self.calibration: CalibrationSnapshot | None = None
        self.field_epoch = 0
        self.field_id = uuid.uuid4().hex
        self.presented: PresentedFrameToken | None = None
        self.latest_frame: FrameHandle | None = None
        self._source_identity = None
        self._latest_frame_lease = None
        self._presented_frame_lease = None
        self._presentation_lease = None
        self._draft_lease = None
        self._presented_context = None
        self._pointer_frames = {}
        self.state = "unavailable"
        self.draft: tuple[PresentedFrameToken, tuple[float, float]] | None = None
        self._draft_context = None
        self.current_records: list[str] = []
        # Identity/state only: never retain a completed intent's full frame.
        self._classification_target = None
        self._classification_generation = 0
        self._measurement_number = 0
        self._evidence_records: dict[str, list[str]] = {}
        self.pending_classification: deque[tuple[MeasurementIntent, dict]] = deque()
        self._confirmation = None
        self._manual = manual_service
        self._manual_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-live-manual")
        self._motion_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-live-motion")
        self.verify_before_present = verify_before_present
        self._prepare_pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-live-prepare")
                              if verify_before_present else None)
        self._preparer = MotionPreparer()
        self._prepare_future = None
        self._motion_item = None
        self._last_prepare_sequence = -1
        self._motion = FieldMotionDetector()
        self._motion_future = None
        self._motion_reset = True
        self._motion_sequence = -1
        self._last_motion_submit = 0.0
        self._stable_count = 0
        self._jobs = deque()
        self._automatic = None
        self._automatic_context = None
        self.candidates = []
        self._candidate_queue = deque()
        self.automatic_enabled = False
        self.automatic_active = False
        self.automatic_prepared = False
        self.automatic_algorithm = "traditional"
        self._automatic_stable_since = None
        self._automatic_cycle_done = False
        self._refresh_pending = None
        self._alignment = CandidateAlignmentMonitor(self.budget, self._request_poll)
        self._candidate_ready_previous = False
        self._candidate_preview_pending = None
        self._candidate_preview_lease = None
        self._candidate_preview_deadline = None
        self._candidate_preview_backoff_until = 0.
        self._candidate_preview_timely_checks = 0
        self._candidate_preview_recovery_since = None
        self._candidate_preview_fallbacks = 0
        self._candidate_presentation_events = deque(maxlen=128)
        self._candidate_pause_started_at = None
        self._candidate_available_started_at = None
        self._candidate_max_pause_ms = 0.
        self._candidate_last_recovery_ms = None
        self._candidate_visibility_previous = None
        self._candidate_visibility_active = False
        self._candidate_visibility_transitions = 0
        self._candidate_visibility_events = deque(maxlen=128)
        self._candidate_visible_since = None
        self._candidate_hidden_since = None
        self._candidate_max_visible_ms = 0.
        self._candidate_max_hidden_ms = 0.
        self._candidate_last_published_sequence = -1
        self._preview_arrived_at = None
        self._preview_interval = 1 / 30
        self._preview_intervals = deque(maxlen=16)
        self._candidate_preview_timer = QTimer(self)
        self._candidate_preview_timer.setSingleShot(True)
        self._candidate_preview_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._candidate_preview_timer.timeout.connect(self._flush_candidate_preview)
        self._performance = LivePerformanceDiagnostics()
        self.candidate_metrics = self._performance.samples
        self._closed = False
        self._polling = False
        self._poll_wake_lock = threading.Lock()
        self._poll_wake_pending = False
        self._completionAvailable.connect(self._drain_poll_wake, Qt.ConnectionType.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setInterval(8)
        self._timer.timeout.connect(self.poll)
        self._timer.start()

    def _request_poll(self, _future=None):
        """Worker-safe, bounded notification; state publication stays on Qt."""
        with self._poll_wake_lock:
            if self._closed or self._poll_wake_pending:
                return
            self._poll_wake_pending = True
        try:
            self._completionAvailable.emit()
        except RuntimeError:
            # A disposed QObject cannot receive a worker completion. Shutdown
            # still joins the pool and releases its frame in _motion_work.
            with self._poll_wake_lock:
                self._poll_wake_pending = False

    @Slot()
    def _drain_poll_wake(self):
        with self._poll_wake_lock:
            self._poll_wake_pending = False
        if not self._closed:
            self.poll()

    def set_calibration(self, value: CalibrationSnapshot | None):
        if value != self.calibration:
            self.end_field("标定已变化")
            self.calibration = value
            self.changed.emit()

    def receive_frame(self, frame: FrameHandle):
        self.observe_source_identity((frame.source_id, frame.generation, frame.size))
        previous = self.latest_frame
        try:
            self._latest_frame_lease = self.budget.replace(self._latest_frame_lease, frame,
                reserve_bytes=self._acquisition_reservation(frame), preview=True)
        except FrameBudgetExceeded:
            # Resource pressure is not field movement. A dropped preview frame
            # must never erase a draft, pending result or accepted overlay.
            self.statusChanged.emit("图像缓冲暂满，已跳过本帧预览并暂停新增测量；当前草稿和视场记录保留。")
            return False
        self.latest_frame = frame
        self._performance.received(frame, time.perf_counter())
        if not frame.live:
            self.state = "ready"
            self._motion_sequence = frame.sequence
        elif previous is None or self.state == "unavailable":
            self.state = "stabilizing"
            self._stable_count = 0
            self._motion_sequence = -1
            self._motion_reset = True
        # Queue compute at arrival instead of waiting for the watchdog tick.
        # This never publishes results or changes which frame the canvas paints.
        self._submit_motion_if_needed()
        return True

    def observe_source_identity(self, identity):
        """Apply source metadata before pixel admission can fail for capacity."""
        if self._source_identity is not None and identity != self._source_identity:
            self.unavailable("采集来源或图像尺寸已变化，正在核对已保存的标尺。")
        self._source_identity = identity

    @staticmethod
    def _acquisition_reservation(frame):
        return acquisition_reservation(frame)

    def _release_frame_reference(self, attribute):
        lease = getattr(self, attribute)
        if lease is not None:
            self.budget.release(lease)
            setattr(self, attribute, None)

    def _acquire_intent_frame(self, frame):
        try:
            return self.budget.acquire(frame, reserve_bytes=self._acquisition_reservation(frame),
                keep_preview_slots=2 if frame.live else 0)
        except FrameBudgetExceeded as exc:
            self._reject(str(exc), code="resource_limit")
            raise

    def feedback_context(self, intent_id=None, *, token=None):
        """Scalar UI context; safe to retain without keeping a frame alive."""
        token = token or self.presented
        frame = token.frame if token is not None else self.latest_frame
        identity = self._source_identity
        return {"intent_id": intent_id, "field_id": self.field_id, "field_epoch": self.field_epoch,
                "source_id": frame.source_id if frame is not None else identity[0] if identity else None,
                "source_generation": frame.generation if frame is not None else identity[1] if identity else None,
                "frame_id": frame.frame_id if frame is not None else None}

    def _context_current(self, token, calibration, operator_id, direction):
        frame = token.frame
        return (not self._closed and token.field_epoch == self.field_epoch
                and calibration == self.calibration
                and operator_id == self.session.active_operator_id
                and direction == self.session.active_direction
                and (self._source_identity is None or
                     self._source_identity == (frame.source_id, frame.generation, frame.size)))

    def _automatic_current(self, context):
        return (self._automatic_context is context and self._context_current(
            context["token"], context["calibration"], context["operator_id"], context["direction"]))

    def _reject(self, reason, *, intent=None, context=None, code=None, severity=None):
        """Publish only a current GUI-owned result; keep the old text signal."""
        def current():
            return (not self._closed and (intent is None or self._valid(intent))
                    and (context is None or self._automatic_current(context)))
        if not current():
            return
        token = intent.token if intent is not None else context["token"] if context else None
        identity = intent.id if intent is not None else str(context["task_id"]) if context else None
        event = feedback_for_reason(reason, code=code, severity=severity,
                                    **self.feedback_context(identity, token=token))
        self.resultRejected.emit(event.message)
        # A compatibility subscriber may synchronously navigate away or start
        # another field. Its old result must not cover the new viewport.
        if current() and event.field_epoch == self.field_epoch:
            self.measurementFeedback.emit(event)

    def mark_presented(self, token: PresentedFrameToken):
        if token.field_epoch == self.field_epoch and self.latest_frame is not None:
            if self.presented is None or self.presented.frame is not token.frame:
                self._release_frame_reference("_presented_frame_lease")
                self._presented_frame_lease = self.budget.acquire(token.frame, preview=True)
            self.presented = token
            self._presented_context = (self.calibration, self.session.active_operator_id,
                                       self.session.active_direction)
            self._release_frame_reference("_presentation_lease")
            self._performance.presented(token.frame, time.perf_counter())

    def present_preview_frame(self, frame):
        """Publish pixels and their verified overlay as one presentation unit.

        The bounded fallback is latched: pixels remain live but the entire
        candidate overlay stays unavailable until sustained timely recovery.
        Individual late completions never make it flash for part of a frame.
        """
        if self._closed:
            return
        now = time.perf_counter()
        if self._preview_arrived_at is not None:
            interval = now - self._preview_arrived_at
            if .004 <= interval <= .2:
                self._preview_intervals.append(interval)
                self._preview_interval = sorted(self._preview_intervals)[len(self._preview_intervals) // 2]
        self._preview_arrived_at = now
        if not self._candidate_preview_guarded() or self.candidates_aligned_for(frame):
            self._clear_candidate_preview()
            self._publish_preview(frame)
            return
        if self._candidate_preview_backoff_until:
            self._clear_candidate_preview()
            self._poll_candidate_alignment()
            if self._candidate_preview_backoff_until or not self._candidate_preview_guarded():
                self._publish_preview(frame)
                return
            # Polling can recover and atomically publish an earlier checked
            # frame. Stage this new input normally instead of immediately
            # covering that verified presentation with unchecked pixels.
        try:
            self._candidate_preview_lease = self.budget.replace(self._candidate_preview_lease, frame, preview=True)
        except FrameBudgetExceeded:
            self._back_off_candidate_preview(now, reason="frame_budget")
            self._clear_candidate_preview()
            self._publish_preview(frame)
            return
        self._candidate_preview_pending = frame
        if self._candidate_preview_deadline is None:
            self._candidate_preview_deadline = now + self._candidate_preview_wait_budget()
            self._candidate_preview_timer.start(max(1, math.ceil((self._candidate_preview_deadline - now) * 1000)))
        self._poll_candidate_alignment()
        self._flush_candidate_preview()

    def should_defer_canvas_business_update(self):
        """A pending live publication will paint business changes within its deadline."""
        frame = self._candidate_preview_pending
        return bool(not self._closed and self.draft is None and frame is not None and frame.live
                    and self._candidate_preview_deadline is not None
                    and not self._candidate_preview_backoff_until and self._candidate_preview_guarded())

    def _candidate_preview_guarded(self):
        return bool(self._automatic_context and self._automatic_context.get("guarded")
                    and any(c.get("state", "available") in {"available", "confirming"} for c in self.candidates))

    def _candidate_preview_wait_budget(self):
        return min(.05, self._preview_interval * .85)

    def _back_off_candidate_preview(self, now, *, reason="verification_late"):
        was_paused = bool(self._candidate_preview_backoff_until)
        self._candidate_preview_backoff_until = now or 1e-12
        if not was_paused:
            self._candidate_preview_fallbacks += 1
            self._candidate_pause_started_at = now
            if self._candidate_available_started_at is not None:
                self._performance.record("candidate_available_ms", (now - self._candidate_available_started_at) * 1000)
            self._candidate_available_started_at = None
            self._candidate_presentation_events.append({"at": now, "state": "paused", "reason": reason})
            self.statusChanged.emit("候选核验中，实时画面继续；核验稳定后恢复点选。")
            self.candidateStateChanged.emit()
        # This is a latch, not a renewable two-second timeout.
        self._candidate_preview_recovery_since = None
        self._candidate_preview_timely_checks = 0

    def _note_candidate_preview_alignment(self, completed, frame):
        if not self._candidate_preview_guarded():
            return
        now = time.perf_counter()
        checked = self._alignment.completed_frame
        submitted_at = self._alignment.completed_submitted_at
        ready_at = self._alignment.completed_ready_at
        elapsed = ((ready_at - submitted_at) * 1000 if ready_at is not None and submitted_at is not None
                   else completed[3].elapsed_ms)
        self._performance.record("candidate_verification_ready_ms", elapsed)
        if ready_at is not None:
            self._performance.record("candidate_verification_gui_wait_ms", (now - ready_at) * 1000)
        if submitted_at is not None:
            self._performance.record("candidate_verification_delivery_ms", (now - submitted_at) * 1000)
        # A result describes its own immutable frame, not whichever input is
        # newest when Qt consumes it. Being one sequence behind is not failure.
        timely = (completed[3].valid and elapsed <= self._candidate_preview_wait_budget() * 1000
                  and checked is not None and self._candidate_frame_fresh(checked, now))
        if not timely:
            reason = ("verification_late" if elapsed > self._candidate_preview_wait_budget() * 1000
                      else "presentation_stale" if checked is not None and not self._candidate_frame_fresh(checked, now)
                      else "alignment_invalid")
            self._back_off_candidate_preview(now, reason=reason)
        elif self._candidate_preview_backoff_until:
            if self._candidate_preview_recovery_since is None:
                self._candidate_preview_recovery_since = now
            self._candidate_preview_timely_checks += 1
            if (now - self._candidate_preview_recovery_since >= .3
                    and self._candidate_preview_timely_checks >= 8):
                self._candidate_preview_backoff_until = 0.
                self._candidate_preview_timely_checks = 0
                self._candidate_preview_recovery_since = None
                if self._candidate_pause_started_at is not None:
                    duration = (now - self._candidate_pause_started_at) * 1000
                    self._performance.record("candidate_unavailable_ms", duration)
                    self._candidate_max_pause_ms = max(self._candidate_max_pause_ms, duration)
                    self._candidate_last_recovery_ms = duration
                self._candidate_pause_started_at = None
                self._candidate_available_started_at = now
                self._candidate_presentation_events.append({"at": now, "state": "ready", "reason": "timely_recovery"})
                self._publish_verified_candidate_frame(checked)
                self.statusChanged.emit("候选已核验，可点选计入或继续手工测量。")
                self.candidateStateChanged.emit()

    def _candidate_frame_fresh(self, frame, now=None):
        # Independent of the 0.85-cycle new-frame wait: a completed exact frame
        # may arrive slightly later through Qt, but never outlive one preview
        # cycle measured from receipt, nor renew its age on subsequent paints.
        return (not frame.live or (0 <= (time.perf_counter() if now is None else now) - frame.received_at
                                 <= self._preview_interval))

    def _publish_verified_candidate_frame(self, frame):
        """Never let a late completion move the visible stream backwards."""
        if (frame is None or self._candidate_preview_backoff_until
                or not self._candidate_frame_fresh(frame)
                or not self._candidate_frame_aligned(frame)
                or frame.sequence < self._candidate_last_published_sequence):
            return False
        context = self._automatic_context
        context["presentation_started"] = True
        if self._candidate_available_started_at is None:
            self._candidate_available_started_at = time.perf_counter()
        self._publish_preview(frame)
        pending = self._candidate_preview_pending
        if pending is not None and frame.sequence >= pending.sequence:
            self._clear_candidate_preview()
        elif pending is not None:
            # Progress was made with an exact verified frame. Give the newer
            # pending frame its own age-bound budget, without extending it on
            # every arrival or timer tick.
            self._candidate_preview_deadline = min(time.perf_counter(), pending.received_at) + self._candidate_preview_wait_budget()
            self._candidate_preview_timer.start(max(1, math.ceil((self._candidate_preview_deadline - time.perf_counter()) * 1000)))
        return True

    def _publish_preview(self, frame):
        self._presentation_lease = self.budget.replace(self._presentation_lease, frame, preview=True)
        self._candidate_last_published_sequence = max(self._candidate_last_published_sequence, frame.sequence)
        self.frameReady.emit(frame)

    def _clear_candidate_preview(self):
        self._candidate_preview_timer.stop()
        self._candidate_preview_pending = None
        self._candidate_preview_deadline = None
        self._release_frame_reference("_candidate_preview_lease")

    def _flush_candidate_preview(self):
        # Qt may deliver the deadline timer before the queued worker wake. Drain
        # an already-ready result first; its own receipt age still gates publish.
        if (not getattr(self, "_flushing_candidate_preview", False) and self._alignment.future is not None
                and self._alignment.future[0].done()):
            self._flushing_candidate_preview = True
            try:
                self._poll_candidate_alignment()
            finally:
                self._flushing_candidate_preview = False
        frame = self._candidate_preview_pending
        if frame is None or self._closed:
            return
        now = time.perf_counter()
        guarded = self._candidate_preview_guarded()
        expired = now >= self._candidate_preview_deadline
        if not guarded or self.candidates_aligned_for(frame) or expired or self._candidate_preview_backoff_until:
            if guarded and expired:
                self._back_off_candidate_preview(now)
            # Acquire presentation ownership before releasing the pending slot.
            self._publish_preview(frame)
            self._clear_candidate_preview()

    def unavailable(self, reason: str):
        self._performance.stop()
        if self.state != "unavailable":
            self.end_field(reason)
        else:
            self.feedbackInvalidated.emit("source_unavailable")
        self.latest_frame = self.presented = None
        self._clear_candidate_preview()
        self._release_frame_reference("_latest_frame_lease")
        self._release_frame_reference("_presented_frame_lease")
        self._release_frame_reference("_presentation_lease")
        self.calibration = None
        self.state = "unavailable"
        self.statusChanged.emit(reason)

    def _cancel_unfinished_measurements(self):
        self.draft = None
        self._draft_context = None
        self._release_frame_reference("_draft_lease")
        for lease in tuple(self._pointer_frames):
            self.release_pointer_frame(lease)
        if self._classification_target and self._classification_target["state"] in {"draft", "pending"}:
            self._classification_target = None
        for intent, _ in self._jobs:
            intent.cancel.set()
        for intent, _ in self.pending_classification:
            self.budget.release(intent.lease)
        self.pending_classification.clear()
        self._cancel_confirmation()
        self._candidate_queue.clear()
        self.candidates = []
        if self._automatic_context:
            if self._automatic is not None:
                self._automatic.cancel()
            self.budget.release(self._automatic_context["lease"])
            self._automatic_context = None
        self._reset_candidate_cycle()

    def cancel_pending_measurements(self, reason="未完成的测量已取消"):
        """Quiesce publication while retaining the accepted current field."""
        self._cancel_unfinished_measurements()
        self.feedbackInvalidated.emit("measurement_cancelled")
        self.candidatesChanged.emit([])
        self.statusChanged.emit(reason)
        self.changed.emit()

    def end_field(self, reason="视场已变化，请重新测量", *, record_end=True):
        if record_end and self.field_id in self.session.fields:
            self.session.update_field(self.field_id, ended_at=datetime.now(timezone.utc).isoformat(), end_reason=reason)
        self.field_epoch += 1
        self.field_id = uuid.uuid4().hex
        self._cancel_unfinished_measurements()
        self.current_records.clear()
        self._classification_target = None
        self._measurement_number = 0
        self._motion_reset = True
        self._motion_sequence = -1
        self._last_prepare_sequence = -1
        self._stable_count = 0
        self.state = "unavailable" if self.latest_frame is None else "stabilizing" if self.latest_frame.live else "ready"
        self.feedbackInvalidated.emit("field_changed")
        self.fieldChanged.emit(self.field_epoch, reason)
        self.candidatesChanged.emit([])
        self.statusChanged.emit(reason)
        self.changed.emit()

    def _pointer_context(self, token):
        retained = next((context for saved, context in self._pointer_frames.values() if saved is token), None)
        if retained is not None:
            return retained
        if token is self.presented:
            return self._presented_context
        return None

    def pointer_token_is_current(self, token):
        """A retained press may outlive preview frames, never its context."""
        if (not isinstance(token, PresentedFrameToken) or self._closed or self.state != "ready"
                or self.presented is None or token.display_revision != self.presented.display_revision):
            return False
        context = self._pointer_context(token)
        return bool(context is not None and self.calibration is not None
                    and self._context_current(token, *context))

    def retain_pointer_frame(self, token):
        if not self.pointer_token_is_current(token):
            raise ValueError("按下时的图像或测量条件已变化，请重新落点。")
        context = self._pointer_context(token)
        lease = self._acquire_intent_frame(token.frame)
        self._pointer_frames[lease] = (token, context)
        return lease

    def release_pointer_frame(self, lease):
        if self._pointer_frames.pop(lease, None) is not None:
            self.budget.release(lease)

    def candidate_interaction_started(self):
        """End manual post-selection ownership before any candidate action.

        Keep an explicit sentinel rather than None: None permits the legacy
        restored-record fallback. A delayed manual completion may still enter
        the ledger, but cannot restore its previous post-selection ownership.
        """
        if self.draft is not None:
            self.cancel_draft()
        self._classification_generation += 1
        self._classification_target = {"id": uuid.uuid4().hex, "number": None,
            "field_epoch": self.field_epoch, "operator_id": self.session.active_operator_id,
            "direction": self.session.active_direction, "state": "candidate",
            "generation": self._classification_generation, "record_id": None}

    def start(self, point: tuple[float, float], *, token=None):
        if self._closed:
            raise RuntimeError("试验工作区已关闭。")
        if (self.state != "ready" or self.presented is None
                or self.presented.field_epoch != self.field_epoch):
            raise ValueError("请等待图像来源和视场稳定。")
        if self.calibration is None:
            raise ValueError("请先设置当前图像来源的标定。")
        token = self.presented if token is None else token
        if not self.pointer_token_is_current(token):
            raise ValueError("按下时的图像或测量条件已变化，请重新落点。")
        self._validate_point(point, token.frame)
        lease = self._acquire_intent_frame(token.frame)
        self._release_frame_reference("_draft_lease")
        self._draft_lease = lease
        self.draft = (token, point)
        self._draft_context = self._pointer_context(token)
        self._measurement_number += 1
        self._classification_generation += 1
        self._classification_target = {"id": uuid.uuid4().hex, "number": self._measurement_number, "field_epoch": self.field_epoch,
            "operator_id": self.session.active_operator_id, "direction": self.session.active_direction,
            "state": "draft", "record_id": None, "generation": self._classification_generation}

    def cancel_draft(self):
        if self.draft is not None and self._classification_target is not None:
            self._classification_target["state"] = "cancelled"
        self.draft = None
        self._draft_context = None
        self._release_frame_reference("_draft_lease")
        self.changed.emit()

    @staticmethod
    def _validate_point(point, frame):
        if not (all(math.isfinite(float(v)) for v in point) and
                0 <= point[0] < frame.size[0] and 0 <= point[1] < frame.size[1]):
            raise ValueError("测量点必须位于图像内。")

    def finish(self, point, material_id, *, method="assisted"):
        if self._closed:
            raise RuntimeError("试验工作区已关闭。")
        if self.draft is None:
            raise ValueError("请先选择起点。")
        token, start = self.draft
        if (token.field_epoch != self.field_epoch or self.state != "ready"
                or self._draft_context is None or not self._context_current(token, *self._draft_context)):
            raise ValueError("视场已变化，请重新测量。")
        self._validate_point(point, token.frame)
        if self.calibration is None:
            raise ValueError("请先标定。")
        if method not in {"assisted", "free"}:
            raise ValueError("未知测量方式。")
        if method == "assisted" and not self.calibration.isotropic:
            raise ValueError("贴边测径需要等比例标定；可主动切换自由测量。")
        if len(self._jobs) + len(self.pending_classification) >= 16:
            raise ValueError("测量队列已满，请等待待处理结果。")
        lease = self._acquire_intent_frame(token.frame)
        self.draft = None
        self._draft_context = None
        self._release_frame_reference("_draft_lease")
        identity = self._classification_target["id"] if self._classification_target else uuid.uuid4().hex
        intent = MeasurementIntent(identity, token, start, tuple(point), self.calibration,
            self.session.active_operator_id, self.session.active_direction, material_id,
            method, threading.Event(), lease, time.perf_counter())
        if self._classification_target is not None:
            self._classification_target["state"] = "pending"
            intent.measurement_number = self._classification_target["number"]
            intent.classification_generation = self._classification_generation
        future = self._manual_pool.submit(self._compute, intent)
        self._jobs.append((intent, future))
        future.add_done_callback(self._request_poll)
        self._submit_motion_if_needed()
        return intent.id

    def _compute(self, intent):
        def check():
            if intent.cancel.is_set():
                raise RuntimeError("测量已取消")
        check()
        if intent.method == "free":
            dx, dy = (intent.end[i] - intent.start[i] for i in range(2))
            width = math.hypot(dx * intent.calibration.x_um, dy * intent.calibration.y_um)
            return {"status": "ok" if width > 0 else "rejected", "reason": "零长度测量" if width <= 0 else "",
                    "width_px": math.hypot(dx, dy), "diameter_um": width,
                    "start": intent.start, "end": intent.end}
        service = self._manual
        if service is None:
            from fdm.services.cu_manual_measurement import measure_manual
            service = measure_manual
        pixels = intent.token.frame.pixels
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            # The kernel accepts strided BGR; its service owns normalization.
            pixels = pixels[:, :, :3]
        result = dict(service(pixels, intent.start, intent.end, checkpoint=check))
        result["diameter_um"] = float(result.get("width_px") or 0) * intent.calibration.x_um
        return result

    def _valid(self, intent):
        return (not intent.cancel.is_set() and self._context_current(intent.token,
                intent.calibration, intent.operator_id, intent.direction))

    def _set_classification_state(self, intent, state, record_id=None):
        target = self._classification_target
        if (target is not None and target["id"] == intent.id
                and intent.classification_generation == self._classification_generation):
            target.update(state=state, record_id=record_id)

    def reclassify_latest(self, material_id):
        """Post-selection follows the latest gesture identity, never an older result."""
        material = self.session.material(material_id)
        target = self._classification_target
        if target is not None:
            if (target["field_epoch"] != self.field_epoch
                    or target["operator_id"] != self.session.active_operator_id
                    or target["direction"] != self.session.active_direction):
                return None
            if target["state"] in {"cancelled", "failed"}:
                self.statusChanged.emit("最近测量已取消或失败；已选择下一条类别，未改动之前的记录。")
                return None
            if target["state"] == "draft":
                return None  # finish receives the currently selected category.
            if target["state"] == "candidate":
                return None  # Candidate clicks always use their own captured class.
            if target["state"] == "pending":
                pairs = list(self._jobs) + list(self.pending_classification)
                request = self._confirmation
                if request and request["kind"] == "manual":
                    pairs += request["intents"]
                for intent, result in pairs:
                    if intent.id != target["id"]:
                        continue
                    intent.material_id = material_id
                    intent.range_confirmation = None
                    if request and request["kind"] == "manual" and request["intents"][0][0] is intent:
                        if self._out_of_range(intent, result):
                            self._confirmation = None
                            self.pending_classification.appendleft((intent, result))
                            self.statusChanged.emit("新类别仍越限，请核对后保留、改类或放弃。")
                    self.changed.emit()
                    return None
                target["state"] = "failed"
                return None
            record = self.session.measurements.get(target["record_id"])
        else:
            # Records restored/rebuilt without an active gesture may still be
            # explicitly post-classified within this field only.
            record = next((record for key in reversed(self.current_records)
                if (record := self.session.measurements.get(key)) is not None
                and record.field_id == self.field_id
                and record.operator_id == self.session.active_operator_id
                and record.direction == self.session.active_direction), None)
        if (record is None or record.field_id != self.field_id
                or record.operator_id != self.session.active_operator_id
                or record.direction != self.session.active_direction):
            return None
        if ((material.lower_limit is not None and record.diameter_um < material.lower_limit)
                or (material.upper_limit is not None and record.diameter_um > material.upper_limit)):
            lower = "不限" if material.lower_limit is None else f"{material.lower_limit:g}"
            upper = "不限" if material.upper_limit is None else f"{material.upper_limit:g}"
            self.statusChanged.emit(f"直径 {record.diameter_um:g} μm 超出 {material.name} 的范围 {lower}–{upper} μm；"
                                    "原记录类别已保留。如确认需要修订，请使用‘改径／改类’明确修改。")
            return None
        self.session.edit_measurement(record.id, material_id=material_id)
        self.changed.emit()
        return record.id

    def _advance_submission_barrier(self, intent, sequence):
        if intent.submission_barrier_at is None:
            intent.submission_barrier_at = time.perf_counter()
            intent.submission_barrier_initial_sequence = sequence
        if intent.submission_barrier_sequence != sequence:
            intent.submission_barrier_previous_sequence = intent.submission_barrier_sequence
            intent.submission_barrier_sequence = sequence

    def _require_current_presented_verified(self, intent):
        if not self._valid(intent) or self.presented is None or self.presented.field_epoch != self.field_epoch:
            raise ValueError("视场已变化，旧结果不能入账。")
        if self.presented.frame.live:
            self._advance_submission_barrier(intent, self.presented.frame.sequence)
            if self._motion_sequence < intent.submission_barrier_sequence:
                raise ValueError("最新显示视场尚未核验，本次结果未入账。")

    def _motion_work(self, holder, reset):
        # Empty the executor's argument holder too: deleting a local frame alone
        # would leave ThreadPoolExecutor._WorkItem.args retaining its pixels.
        frame, lease = holder
        holder.clear()
        try:
            key = (frame.source_id, frame.generation, frame.size)
            frame_id, sequence = frame.frame_id, frame.sequence
            gray = reduced_gray(frame.pixels)
        finally:
            self.budget.release(lease)
            del frame
        if reset:
            self._motion.reset()
        # poll() ends the field on instability and the next job resets the
        # detector. Building a reference here would be discarded immediately.
        return self._motion.compare_reduced(gray, key, frame_id, sequence)

    def _submit_motion_if_needed(self):
        if self._closed:
            return
        if self.verify_before_present:
            self._advance_motion_pipeline()
            return
        frame = self.latest_frame
        if frame and frame.live and self._motion_future is None and frame.sequence > self._motion_sequence:
            now = time.perf_counter()
            if now - self._last_motion_submit >= 0.04 or self._jobs or self._confirmation is not None:
                try:
                    lease = self.budget.acquire(frame)
                except FrameBudgetExceeded:
                    if self.state != "stabilizing":
                        self.state = "stabilizing"
                        self.statusChanged.emit("图像缓冲已满，暂停新测量并等待当前任务释放。")
                else:
                    try:
                        future = self._motion_pool.submit(self._motion_work, [frame, lease], self._motion_reset)
                    except Exception:
                        self.budget.release(lease)
                        raise
                    self._motion_future = (self.field_epoch, future)
                    self._motion_reset = False
                    self._last_motion_submit = now
                    future.add_done_callback(self._request_poll)

    def _release_motion_item(self, item):
        if item is not None:
            with item.lock:
                item.armed = False
                if item.lease is not None:
                    self.budget.release(item.lease)
                    item.lease = None
                item.frame = item.prepared = None

    @staticmethod
    def _submit_pipeline(pool, function, item, *args):
        # ThreadPoolExecutor may enqueue a WorkItem before Thread.start fails.
        # A rejected submission must leave that residual work inert.
        with item.lock:
            item.armed = False
            future = pool.submit(function, item, *args)
            item.armed = True
            return future

    def _prepare_motion_work(self, item):
        with item.lock:
            if not item.armed:
                return None
            frame = item.frame
        return self._preparer.prepare(frame)

    def _prepared_motion_work(self, item, reset):
        with item.lock:
            if not item.armed:
                return None
            frame, prepared = item.frame, item.prepared
        if reset:
            self._motion.reset()
        return self._motion.compare_prepared(prepared,
            (frame.source_id, frame.generation, frame.size), frame.frame_id, frame.sequence)

    def _advance_motion_pipeline(self):
        """One preparation slot, one ordered decision slot, newest arrival only.

        Neither worker publishes Qt state. Every in-flight raw frame has its
        own budget lease, including the handoff to a not-yet-painted canvas.
        """
        if self._prepare_future is not None:
            epoch, future, item = self._prepare_future
            if future.done() and (epoch != self.field_epoch or self._motion_future is None):
                self._prepare_future = None
                try:
                    prepared = future.result()
                    if epoch == self.field_epoch:
                        item.prepared = prepared
                        decision = self._submit_pipeline(self._motion_pool, self._prepared_motion_work,
                                                         item, self._motion_reset)
                        self._motion_future = (epoch, decision)
                        self._motion_item = item
                        self._motion_reset = False
                        item = None  # Decision now owns the lease.
                        decision.add_done_callback(self._request_poll)
                except Exception as exc:
                    if epoch == self.field_epoch:
                        self.unavailable(f"视场检测失败：{exc}")
                    exc.__traceback__ = None
                finally:
                    self._release_motion_item(item)
        frame = self.latest_frame
        if (frame is None or not frame.live or self._prepare_future is not None
                or frame.sequence <= self._last_prepare_sequence):
            return
        # Bound high-rate sources, without the old 40 ms idle gap reducing a
        # 30 Hz stream to alternating display frames. No input queue accumulates.
        now = time.perf_counter()
        if now - self._last_motion_submit < .025:
            return
        try:
            item = _MotionFrame(frame, self.budget.acquire(frame))
        except FrameBudgetExceeded:
            self.statusChanged.emit("图像缓冲已满，等待当前任务释放；当前草稿和视场记录保留。")
            return
        try:
            future = self._submit_pipeline(self._prepare_pool, self._prepare_motion_work, item)
        except Exception as exc:
            self._release_motion_item(item)
            self.unavailable(f"视场检测无法启动：{exc}")
            exc.__traceback__ = None
            return
        self._prepare_future = (self.field_epoch, future, item)
        self._last_prepare_sequence = frame.sequence
        self._last_motion_submit = now
        future.add_done_callback(self._request_poll)

    def _present_motion_frame(self, item):
        # Completion is handled on Qt before any record can enter the ledger.
        # A moving frame is also displayed, after end_field has cleared markers;
        # two stable decisions are still required before accepting new gestures.
        self.present_preview_frame(item.frame)

    def poll(self):
        if self._closed or self._polling:
            return
        self._polling = True
        try:
            self._poll_once()
        finally:
            self._polling = False

    def _poll_once(self):
        if self._motion_future and self._motion_future[1].done():
            epoch, future = self._motion_future
            self._motion_future = None
            item, self._motion_item = self._motion_item, None
            try:
                result = future.result()
                if epoch == self.field_epoch:
                    if not result.stable:
                        self.end_field("视场已变化，请重新测量")
                    else:
                        self._motion_sequence = result.sequence
                        self._stable_count += 1
                        if self._stable_count >= 2:
                            if self.state != "ready":
                                self.statusChanged.emit("视场稳定，可以测量")
                            self.state = "ready"
                    if item is not None:
                        self._present_motion_frame(item)
            except Exception as exc:
                if epoch == self.field_epoch:
                    self.unavailable(f"视场检测失败：{exc}")
                exc.__traceback__ = None
            finally:
                self._release_motion_item(item)
        self._submit_motion_if_needed()
        # Publish completed verification before synchronous ledger/gallery work.
        self._poll_candidate_alignment()
        deadline = time.perf_counter() + 0.004
        while self._jobs and self._jobs[0][1].done() and time.perf_counter() < deadline:
            intent, future = self._jobs[0]
            if self._valid(intent) and self.presented and self.presented.frame.live:
                # Every publication must cover the currently displayed frame.
                # Advance the target, but never restart the original timeout.
                self._advance_submission_barrier(intent, self.presented.frame.sequence)
                elapsed = time.perf_counter() - intent.submission_barrier_at
                if elapsed > 2:
                    self._reject("verification_timeout", intent=intent)
                    intent.cancel.set()
                elif self._motion_sequence < intent.submission_barrier_sequence:
                    if elapsed > .5 and self.state == "ready":
                        self.state = "stabilizing"
                        self.statusChanged.emit("视场核验较慢，正在暂停新测量并等待当前结果。")
                    break
            self._jobs.popleft()
            keep_lease = False
            try:
                if future.cancelled():
                    self._set_classification_state(intent, "cancelled")
                    continue
                if not self._valid(intent):
                    continue
                result = future.result()
                if result.get("status") not in {"ok", "success", "valid"}:
                    self._reject(result.get("reason") or "edge_not_found", intent=intent)
                    continue
                if not (math.isfinite(float(result.get("diameter_um", 0))) and result["diameter_um"] > 0):
                    self._reject("invalid_width", intent=intent)
                    continue
                if self._out_of_range(intent, result):
                    self.pending_classification.append((intent, result))
                    keep_lease = True
                    self.statusChanged.emit("测量超出类别范围，请在待确认项中保留、改类或放弃。")
                    self.changed.emit()
                else:
                    self._accept(intent, result)
            except Exception as exc:
                self._reject(str(exc), intent=intent, severity="error")
            finally:
                if not keep_lease:
                    target = self._classification_target
                    if target is not None and target["id"] == intent.id and target["state"] == "pending":
                        self._set_classification_state(intent, "failed")
                    self.budget.release(intent.lease)
        self._poll_confirmation()
        self._advance_candidate_queue()
        for entry in self.evidence.poll():
            for record_id in self._evidence_records.pop(entry.get("evidence_id", entry["frame_id"]), []):
                self.session.set_evidence(record_id, dict(entry))
            self.changed.emit()
        self._poll_automatic()
        self._poll_candidate_alignment()
        self._schedule_automatic()

    def _out_of_range(self, intent, result):
        operator = self.session.operators[intent.operator_id]
        material = next((m for m in operator.materials[intent.direction] if m.id == intent.material_id), None)
        if material is None:
            raise ValueError("请选择有效纤维类别。")
        lower = material.lower_limit
        upper = material.upper_limit
        value = result["diameter_um"]
        return ((lower is not None and lower >= 0 and value < lower) or
                (upper is not None and upper >= 0 and value > upper))

    def _range_signature(self, intent, result):
        material = self.session.material(intent.material_id, operator_id=intent.operator_id, direction=intent.direction)
        return {"material_id": material.id, "material_name": material.name,
                "lower_limit_um": material.lower_limit, "upper_limit_um": material.upper_limit,
                "diameter_um": result["diameter_um"]}

    def _validate_range_confirmation(self, intent, result):
        if intent.range_confirmation is not None:
            if intent.range_confirmation != self._range_signature(intent, result):
                raise ValueError("类别或范围已变化，请重新核对越限确认。")
        elif self._out_of_range(intent, result):
            raise ValueError("存在尚未确认保留的越限结果，请先处理。")

    def _accept(self, intent, result):
        self._require_current_presented_verified(intent)
        self._validate_range_confirmation(intent, result)
        evidence = self._reserve_evidence(intent, result)
        record = self.session.add_measurement(**self._entry(intent, result, evidence))
        self._ensure_field(intent, record.created_at)
        self.current_records.append(record.id)
        self._set_classification_state(intent, "accepted", record.id)
        if evidence.get("state") == "pending":
            self._evidence_records.setdefault(evidence.get("evidence_id", intent.token.frame.frame_id), []).append(record.id)
        self.recordAccepted.emit(record)
        self.changed.emit()
        return record

    def _reserve_evidence(self, intent, result):
        if not self.session.save_images:
            return {"state": "disabled"}
        options = getattr(self.session, "evidence_options", None)
        if options is None:
            # Historical sessions retain complete PNG semantics and legacy
            # EvidenceStore adapters retain their single-argument contract.
            return dict(self.evidence.reserve(intent.token.frame))
        points = measurement_evidence_points({**result, "input_start": intent.start, "input_end": intent.end})
        return dict(self.evidence.reserve(intent.token.frame, options=options, points=points))

    def _ensure_field(self, intent, created_at):
        if self.field_id not in self.session.fields:
            frame = intent.token.frame
            self.session.update_field(self.field_id, opened_at=created_at,
                source_id=frame.source_id, source_generation=frame.generation,
                first_frame_id=frame.frame_id, first_frame_received_at=frame.received_at,
                operator_id=intent.operator_id, direction=intent.direction,
                calibration={"x_um": intent.calibration.x_um, "y_um": intent.calibration.y_um,
                             "revision": intent.calibration.revision, "source": intent.calibration.source})

    def _entry(self, intent, result, evidence):
        return dict(diameter_um=result["diameter_um"], material_id=intent.material_id,
            operator_id=intent.operator_id, direction=intent.direction,
            field_id=self.field_id, frame_id=intent.token.frame.frame_id,
            endpoints=[list(result["start"]), list(result["end"])], evidence=evidence,
            measurement_method=intent.method, provenance={
                "algorithm_revision": result.get("revision", "cu-manual-portable-v1" if intent.method == "assisted" else "free-v1"),
                "motion_revision": MOTION_REVISION, "width_px": result["width_px"],
                "original_diameter_um": result["diameter_um"],
                "calibration": {"x_um": intent.calibration.x_um, "y_um": intent.calibration.y_um,
                                "revision": intent.calibration.revision, "source": intent.calibration.source},
                "frame_metadata": intent.token.frame.metadata,
                "input_start": list(intent.start), "input_end": list(intent.end),
                "diagnostics": result.get("diagnostics", {}),
                "received_at": intent.token.frame.received_at,
                "submission_barrier_sequence": intent.submission_barrier_sequence,
                "submission_barrier_initial_sequence": intent.submission_barrier_initial_sequence,
                "submission_barrier_previous_sequence": intent.submission_barrier_previous_sequence,
                "measurement_operation_id": intent.id,
                "measurement_number": intent.measurement_number,
                "range_confirmation": deepcopy(intent.range_confirmation),
                "latency_ms": (time.perf_counter() - intent.submitted_at) * 1000})

    def resolve_pending(self, *, material_id=None, discard=False, intent_id=None):
        if self._confirmation is not None:
            raise ValueError("正在核验上次确认时的视场，请等待结果。")
        if not self.pending_classification:
            if intent_id is not None:
                raise ValueError("该待确认测量已失效，请核对当前提示。")
            return
        intent, result = self.pending_classification[0]
        if intent_id is not None and intent.id != intent_id:
            raise ValueError("待确认项已变化，本次操作未应用；请核对当前提示后重试。")
        if material_id and material_id != intent.material_id:
            intent.material_id = material_id
            if self._out_of_range(intent, result):
                self.statusChanged.emit("仍超出新类别范围；如确认无误，请再次选择保留。")
                self.changed.emit()
                return
        if discard:
            self.pending_classification.popleft()
            self._set_classification_state(intent, "cancelled")
            self.budget.release(intent.lease)
            self.changed.emit()
            return
        if not self._valid(intent) or self.presented is None:
            raise ValueError("视场已变化，旧结果不能入账。")
        intent.range_confirmation = self._range_signature(intent, result) if self._out_of_range(intent, result) else None
        self.pending_classification.popleft()
        self._queue_confirmation("manual", [(intent, deepcopy(result))])

    def _queue_confirmation(self, kind, intents, *, context=None, candidates=None):
        # Each user action gets a fresh timeout. Its target follows the latest
        # displayed frame until verified; following frames never extend time.
        sequence = self.presented.frame.sequence if self.presented.frame.live else None
        started = time.perf_counter()
        for intent, _ in intents:
            intent.submission_barrier_sequence = sequence
            intent.submission_barrier_at = started
            intent.submission_barrier_initial_sequence = sequence
            intent.submission_barrier_previous_sequence = None
        self._confirmation = {"kind": kind, "intents": intents, "context": context,
            "candidates": candidates, "sequence": sequence, "started": started}
        if sequence is not None and self._motion_sequence < sequence:
            self.statusChanged.emit("正在核验确认时的视场，核验完成后自动入账。")
        self.changed.emit()
        self._submit_motion_if_needed()
        self._poll_confirmation()

    def _cancel_confirmation(self):
        request, self._confirmation = self._confirmation, None
        if request and request["kind"] == "manual":
            for intent, _ in request["intents"]:
                self._set_classification_state(intent, "cancelled")
                self.budget.release(intent.lease)
        elif request:
            self._release_candidate_request(request)

    def _poll_confirmation(self):
        request = self._confirmation
        if request is None:
            return
        valid = all(self._valid(intent) for intent, _ in request["intents"])
        if request["kind"] == "automatic":
            valid = valid and self._automatic_context is request["context"]
        if not valid or self.presented is None:
            self._cancel_confirmation()
            # Invalidated work is silent: field/source transitions already
            # dismiss feedback, and old confirmations cannot describe a new field.
            self.changed.emit()
            return
        sequence = self.presented.frame.sequence if self.presented.frame.live else None
        request["sequence"] = sequence
        for intent, _ in request["intents"]:
            self._advance_submission_barrier(intent, sequence)
        elapsed = time.perf_counter() - request["started"]
        if sequence is not None and elapsed > 2:
            self._reject("confirmation_timeout", intent=request["intents"][0][0])
            self._cancel_confirmation()
            self.changed.emit()
            return
        if sequence is not None and self._motion_sequence < sequence:
            if elapsed > .5 and self.state == "ready":
                self.state = "stabilizing"
                self.statusChanged.emit("视场核验较慢，正在暂停新测量并等待本次确认。")
            return
        if request["kind"] == "automatic" and not self.candidate_overlay_ready:
            # Alignment work is asynchronous and must cover the presented
            # frame. Keep the same deadline while waiting; never move markers.
            return
        # Remove the pending request before synchronous UI callbacks run. Its
        # lease remains held until the complete domain transaction has finished.
        self._confirmation = None
        try:
            if request["kind"] == "manual":
                self._accept(*request["intents"][0])
            else:
                self._commit_candidates(request["context"], request["intents"], request["candidates"])
        except Exception as exc:
            if request["kind"] == "manual":
                self._set_classification_state(request["intents"][0][0], "failed")
            self._reject(str(exc), intent=request["intents"][0][0], severity="error")
        finally:
            if request["kind"] == "manual":
                for intent, _ in request["intents"]:
                    self.budget.release(intent.lease)
            else:
                self._release_candidate_request(request)
            self.changed.emit()

    def configure_automatic(self, *, enabled=None, active=None, prepared=None, algorithm=None):
        """Application preferences/readiness; this never edits the session."""
        if self._closed:
            return
        before = (self.automatic_enabled, self.automatic_active, self.automatic_prepared, self.automatic_algorithm)
        was_active = self.automatic_active
        if algorithm is not None:
            if algorithm not in {"traditional", "multiscale"}:
                raise ValueError("未知的全图分析算法。")
            self.automatic_algorithm = algorithm
        if enabled is not None:
            self.automatic_enabled = bool(enabled)
        if active is not None:
            self.automatic_active = bool(active)
        if prepared is not None:
            self.automatic_prepared = bool(prepared)
        context = self._automatic_context
        if not self.automatic_enabled and context and context.get("origin") == "automatic" and not context.get("completed"):
            self.discard_candidates()
        if was_active and not self.automatic_active:
            self.cancel_candidate_submissions()
            self.discard_candidates()
            self._reset_candidate_cycle()
        after = (self.automatic_enabled, self.automatic_active, self.automatic_prepared, self.automatic_algorithm)
        if after != before:
            self.candidateStateChanged.emit()

    def performance_diagnostics(self):
        """Bounded scalar timings, suitable for offline diagnostic export."""
        return self._performance.summary()

    def performance_diagnostics_schema(self):
        return self._performance.schema()

    def performance_frame_context(self):
        context = self._performance.frame_context()
        now = time.perf_counter()
        pause_ms = ((now - self._candidate_pause_started_at) * 1000
                    if self._candidate_pause_started_at is not None else None)
        visible_ms = ((now - self._candidate_visible_since) * 1000
                      if self._candidate_visible_since is not None else None)
        hidden_ms = ((now - self._candidate_hidden_since) * 1000
                     if self._candidate_hidden_since is not None else None)
        context["candidate_preview"] = {
            "fallback_count": self._candidate_preview_fallbacks,
            "short_wait_budget_ms": self._candidate_preview_wait_budget() * 1000,
            "publishable_frame_age_ms": self._preview_interval * 1000,
            "readiness_definition": "Submit to parent worker/IPC readiness; Qt consumption delay is recorded separately.",
            "worker_runtime": dict(getattr(self._alignment.worker, "runtime", {})),
            "immediate_mode": bool(self._candidate_preview_backoff_until),
            "recovery_timely_checks": self._candidate_preview_timely_checks,
            "state": "paused" if self._candidate_preview_backoff_until else "coherent",
            "events": list(self._candidate_presentation_events),
            "ongoing_unavailable_ms": pause_ms,
            "max_pause_ms": max(self._candidate_max_pause_ms, pause_ms or 0.),
            "last_recovery_ms": self._candidate_last_recovery_ms,
            "recovery_healthy_ms": ((now - self._candidate_preview_recovery_since) * 1000
                                    if self._candidate_preview_recovery_since is not None else None),
            "painted_visibility": {"visible": self._candidate_visibility_previous if self._candidate_visibility_active else None,
                "last_observed_visible": self._candidate_visibility_previous,
                "observation_active": self._candidate_visibility_active,
                "transition_count": self._candidate_visibility_transitions,
                "current_visible_ms": visible_ms, "current_hidden_ms": hidden_ms,
                "max_visible_ms": max(self._candidate_max_visible_ms, visible_ms or 0.),
                "max_hidden_ms": max(self._candidate_max_hidden_ms, hidden_ms or 0.),
                "events": list(self._candidate_visibility_events),
                "definition": "Canvas paint completion observations; not sensor time or click success. Hidden-duration tracking excludes user-hidden overlays and absent/unaccepted-empty batches."},
            "definition": "A missed presentation budget latches all candidate overlays and hit targets off while pixels remain live. Recovery requires at least 300 ms and 8 consecutive timely exact-frame checks. Verified pixels and overlays are then published together; a valid older completion is not mistaken for a new frame."}
        return context

    def record_candidate_visibility(self, frame, visible, *, display_enabled=True):
        """Observe actual paints without scheduling another paint or UI update."""
        now = time.perf_counter()
        visible = bool(visible)
        was_observing = self._candidate_visibility_active
        self._candidate_visibility_active = True
        active = bool(display_enabled and self._candidate_preview_guarded())
        if visible != self._candidate_visibility_previous:
            if self._candidate_visibility_previous is not None:
                self._candidate_visibility_transitions += 1
            if self._candidate_visible_since is not None:
                self._candidate_max_visible_ms = max(self._candidate_max_visible_ms,
                                                     (now - self._candidate_visible_since) * 1000)
            if self._candidate_hidden_since is not None:
                self._candidate_max_hidden_ms = max(self._candidate_max_hidden_ms,
                                                    (now - self._candidate_hidden_since) * 1000)
            self._candidate_visible_since = now if visible else None
            self._candidate_hidden_since = now if not visible and active else None
            self._candidate_visibility_previous = visible
            reason = ("ready" if visible else "user_hidden" if not display_enabled else
                      "verification_paused" if self._candidate_preview_backoff_until else
                      "pending_verification" if active else "no_candidates")
            context = self._automatic_context
            self._candidate_visibility_events.append({"at": now, "visible": visible, "reason": reason,
                "frame_id": frame.frame_id if frame is not None else None,
                "batch_id": context.get("batch_id") if context else None})
        elif visible and not was_observing:
            self._candidate_visible_since = now
        elif not visible:
            if active and self._candidate_hidden_since is None:
                self._candidate_hidden_since = now
            elif not active and self._candidate_hidden_since is not None:
                self._candidate_max_hidden_ms = max(self._candidate_max_hidden_ms,
                                                    (now - self._candidate_hidden_since) * 1000)
                self._candidate_hidden_since = None

    def _suspend_candidate_visibility(self):
        """Stop elapsed timers on a closed/reset viewport without inventing a paint."""
        now = time.perf_counter()
        if self._candidate_visible_since is not None:
            self._candidate_max_visible_ms = max(self._candidate_max_visible_ms,
                                                 (now - self._candidate_visible_since) * 1000)
        if self._candidate_hidden_since is not None:
            self._candidate_max_hidden_ms = max(self._candidate_max_hidden_ms,
                                                (now - self._candidate_hidden_since) * 1000)
        self._candidate_visible_since = self._candidate_hidden_since = None
        self._candidate_visibility_active = False

    def record_candidate_painted(self, frame):
        context = self._automatic_context
        if (not context or context.get("painted_at") is not None or not context.get("completed")
                or not self.candidates or not self.candidates_aligned_for(frame)):
            return
        context["painted_at"] = time.perf_counter()
        if context.get("started_at") is not None:
            self.candidate_metrics["analysis_to_paint_ms"].append(
                (context["painted_at"] - context["started_at"]) * 1000)

    @property
    def candidate_overlay_ready(self):
        return self.candidates_aligned_for(self.presented.frame if self.presented else None)

    def candidates_aligned_for(self, frame):
        """Paint validates the frame it is about to show, not its previous token."""
        if self._candidate_preview_backoff_until:
            return False
        context = self._automatic_context
        if context and not context.get("presentation_started", True):
            return False
        return self._candidate_frame_aligned(frame)

    def _candidate_frame_aligned(self, frame):
        """Exact-frame fact, independent of the latched presentation state."""
        context = self._automatic_context
        if not context or frame is None or not self._automatic_current(context):
            return False
        if not context.get("guarded"):
            # Older integrations/tests can inject an already-reviewed batch.
            return True
        result = self._alignment.result_for(frame)
        return bool(result and result[3].valid and self._alignment.reference is context["token"].frame
                    and result[0] == self._source_identity
                    and result[0] == (frame.source_id, frame.generation, frame.size)
                    and result[2] == frame.frame_id)

    @property
    def candidate_pending_count(self):
        return sum(len(item["intents"]) for item in self._candidate_queue) + (
            len(self._confirmation["intents"]) if self._confirmation and self._confirmation["kind"] == "automatic" else 0)

    @property
    def has_unaccepted_candidates(self):
        return bool(self.candidate_pending_count or any(c.get("state", "available") == "available" for c in self.candidates)
                    or self._automatic_context is not None and not self._automatic_context.get("completed", False))

    def _reset_candidate_cycle(self):
        self._suspend_candidate_visibility()
        if self._candidate_pause_started_at is not None:
            self._candidate_max_pause_ms = max(self._candidate_max_pause_ms,
                (time.perf_counter() - self._candidate_pause_started_at) * 1000)
        self._alignment.reset()
        self._clear_candidate_preview()
        self._candidate_preview_backoff_until = 0.
        self._candidate_preview_timely_checks = 0
        self._candidate_preview_recovery_since = None
        self._candidate_pause_started_at = None
        self._candidate_available_started_at = None
        self._candidate_last_published_sequence = -1
        self._automatic_stable_since = None
        self._automatic_cycle_done = False
        self._refresh_pending = None

    def _poll_candidate_alignment(self):
        if self._automatic_context is not None and not self._automatic_context.get("guarded"):
            return
        needed = self._automatic_context is not None or (
            self.automatic_enabled and self.automatic_active and self.automatic_prepared)
        frame = self.latest_frame
        if not needed or frame is None:
            if self._alignment.reference is not None:
                self._alignment.reset()
            return
        if self._alignment.reference is None:
            try:
                self._alignment.reset(frame)
            except FrameBudgetExceeded:
                return
        completed = self._alignment.poll(frame)
        if completed is not None:
            self.candidate_metrics["alignment_ms"].append(completed[3].elapsed_ms)
            self._note_candidate_preview_alignment(completed, frame)
            if completed[3].valid:
                if self._candidate_preview_guarded() and not self._candidate_preview_backoff_until:
                    self._publish_verified_candidate_frame(self._alignment.completed_frame)
                if self.state == "ready" and self._automatic_stable_since is None:
                    self._automatic_stable_since = time.perf_counter()
            else:
                # Candidate geometry has its own lifecycle. Do not end the
                # business field or erase accepted records on a small drift.
                had_candidates = bool(self.candidates)
                if self._automatic_context is not None:
                    self.discard_candidates()
                self._automatic_cycle_done = False
                self._automatic_stable_since = None
                try:
                    self._alignment.reset(frame)
                    if completed[3].reason in {"low_texture", "alignment_error", "frame_shape"}:
                        # A retry on the identical invalid pixels cannot help.
                        # Wait for a genuinely new source frame, without a loop.
                        self._alignment.last_submitted = frame.frame_id
                except FrameBudgetExceeded:
                    pass
                if had_candidates:
                    self.statusChanged.emit("画面与候选源帧已不一致，旧候选已隐藏；稳定后可重新分析。")
        if (not frame.live and self._candidate_preview_backoff_until
                and self._alignment.future is None):
            # A static image has no next capture tick. Recheck the same exact
            # pixels only during recovery, then return to once-per-frame work.
            self._alignment.last_submitted = None
        if self.state != "ready":
            self._automatic_stable_since = None
        ready = self.candidate_overlay_ready
        if ready != self._candidate_ready_previous:
            self._candidate_ready_previous = ready
            self.candidateStateChanged.emit()
        self._flush_candidate_preview()

    def _schedule_automatic(self):
        if self._refresh_pending is not None and not self.candidate_pending_count:
            algorithm, self._refresh_pending = self._refresh_pending, None
            try:
                self.analyze(algorithm)
            except (ValueError, FrameBudgetExceeded) as exc:
                self.statusChanged.emit(str(exc))
            return
        if (not self.automatic_enabled or not self.automatic_active or not self.automatic_prepared
                or self._automatic_cycle_done or self._automatic_context is not None
                or self.state != "ready" or self.presented is None or self.calibration is None
                or not self.presented.frame.live
                or not self.calibration.isotropic or self._automatic_stable_since is None
                or time.perf_counter() - self._automatic_stable_since < .3
                or self._jobs or self.draft or self.pending_classification or self._confirmation):
            return
        aligned = self._alignment.result_for(self.presented.frame)
        if not aligned or not aligned[3].valid:
            return
        self._automatic_cycle_done = True
        self.candidate_metrics["stable_wait_ms"].append((time.perf_counter() - self._automatic_stable_since) * 1000)
        try:
            self.analyze(self.automatic_algorithm, _origin="automatic")
        except Exception as exc:
            self.statusChanged.emit(f"自动分析无法启动：{exc}。可使用全图分析重试。")

    def analyze(self, algorithm=None, *, _origin="manual"):
        if self._closed:
            raise RuntimeError("试验工作区已关闭。")
        if self.state != "ready" or self.presented is None or self.calibration is None:
            raise ValueError("请等待视场稳定并完成标定。")
        if not self.calibration.isotropic:
            raise ValueError("全图测径需要等比例标定。")
        algorithm = algorithm or self.automatic_algorithm
        if algorithm not in {"traditional", "multiscale"}:
            raise ValueError("未知的全图分析算法。")
        if self.candidate_pending_count:
            self._refresh_pending = algorithm
            self.statusChanged.emit("正在完成已点击的候选，完成后刷新分析。")
            return
        if self._automatic_context is not None:
            self.discard_candidates()
        from fdm.services.cu_automatic_worker import CuAutomaticWorker
        if self._automatic is None:
            self._automatic = CuAutomaticWorker()
        token = self.presented
        lease = self._acquire_intent_frame(token.frame)
        started = time.perf_counter()
        try:
            self._alignment.reset(token.frame)
            task_id = self._automatic.submit(token.frame.pixels, algorithm=algorithm)
        except Exception:
            self.budget.release(lease)
            self._alignment.reset()
            raise
        self._automatic_context = {"task_id": task_id, "token": token, "lease": lease,
            "calibration": self.calibration, "algorithm": algorithm,
            "operator_id": self.session.active_operator_id, "direction": self.session.active_direction,
            "batch_id": uuid.uuid4().hex, "guarded": True, "origin": _origin,
            "completed": False, "started_at": started, "presentation_started": False}
        self._automatic_cycle_done = True
        # The analysis frame becomes the immutable alignment reference, even
        # when the scheduling cycle previously used a slightly earlier frame.
        self._alignment.poll(self.latest_frame)
        self.candidateStateChanged.emit()
        self.statusChanged.emit("正在分析当前视场；实时画面继续更新。")

    def _poll_automatic(self):
        if self._automatic is None:
            return
        for response in self._automatic.poll():
            context = self._automatic_context
            if not context or response.get("task_id") != context["task_id"]:
                continue
            result = response.get("result") or {}
            if not self._automatic_current(context):
                self.discard_candidates()
            elif result.get("status") == "cancelled":
                self.discard_candidates()
            elif result.get("status") in {"ok", "success", "empty"}:
                self._performance.analysis(response)
                context["diagnostics"] = deepcopy(result.get("diagnostics", {}))
                if response.get("timings"):
                    context["diagnostics"]["worker_timings"] = deepcopy(response["timings"])
                context["completed"] = True
                context["latency_ms"] = (time.perf_counter() - context["started_at"]) * 1000
                self.candidate_metrics["analysis_ms"].append(context["latency_ms"])
                self.candidates = list(result.get("candidates", []))
                self._ensure_candidate_ids()
                self.candidatesChanged.emit(self.candidates)
                self.statusChanged.emit(f"检测到 {len(self.candidates)} 条候选，请赋类后确认。" if self.candidates else "本视场未发现候选，可手工测量或刷新")
                # Empty analysis is a quiet status, not failed measurement
                # feedback. Retain its reference so this stable cycle runs once.
            else:
                self._reject(result.get("reason") or response.get("error") or "algorithm_error",
                             context=context, severity="error")
                self.discard_candidates()

    def discard_candidates(self):
        self.cancel_candidate_submissions()
        if self._confirmation and self._confirmation["kind"] == "automatic":
            self._cancel_confirmation()
        if self._automatic_context:
            if self._automatic is not None:
                self._automatic.cancel()
            self.budget.release(self._automatic_context["lease"])
        self._automatic_context = None
        self.candidates = []
        if self._candidate_pause_started_at is not None:
            self._candidate_max_pause_ms = max(self._candidate_max_pause_ms,
                (time.perf_counter() - self._candidate_pause_started_at) * 1000)
        self._clear_candidate_preview()
        self._candidate_preview_backoff_until = 0.
        self._candidate_preview_timely_checks = 0
        self._candidate_preview_recovery_since = None
        self._candidate_pause_started_at = None
        self._candidate_available_started_at = None
        self.candidatesChanged.emit([])
        self.candidateStateChanged.emit()

    def _ensure_candidate_ids(self):
        context = self._automatic_context
        batch = context.setdefault("batch_id", uuid.uuid4().hex) if context else "detached"
        for index, candidate in enumerate(self.candidates):
            if "candidate_id" not in candidate:
                candidate = self.candidates[index] = deepcopy(candidate)
            candidate.setdefault("candidate_id", f"{batch}:{index + 1}")
            candidate.setdefault("ordinal", index + 1)
            candidate.setdefault("state", "available")
            candidate.setdefault("record_id", None)

    def sync_candidate_records(self):
        """Undo/delete expose a candidate again; redo restores its record link."""
        if self.candidate_pending_count:
            raise ValueError("候选正在核验，请等待完成或按 Esc 取消后再撤销。")
        changed = False
        for candidate in self.candidates:
            record_id = candidate.get("record_id")
            if record_id:
                state = ("accepted" if record_id in self.session.measurements else
                         "available" if candidate.get("state") == "accepted" else candidate.get("state", "available"))
                if candidate.get("state") != state:
                    candidate["state"] = state
                    changed = True
        if changed:
            self.candidatesChanged.emit(self.candidates)
        self.candidateStateChanged.emit()

    def submit_candidate(self, candidate_id, material_id):
        """Point selection snapshots the category now; it never auto-classifies."""
        self._ensure_candidate_ids()
        index = next((i for i, c in enumerate(self.candidates) if c["candidate_id"] == candidate_id), None)
        if index is None:
            raise ValueError("该候选已失效，请刷新全图分析。")
        if self.possible_duplicate_candidates([index]):
            raise ValueError("该候选可能与已有记录重复，请在审核页核对后确认。")
        return self.accept_candidates([(index, material_id)], _queue=True)

    def set_candidate_excluded(self, candidate_id, excluded):
        """Exclude only uncommitted geometry; review state is not project data."""
        self._ensure_candidate_ids()
        candidate = next((c for c in self.candidates if c["candidate_id"] == candidate_id), None)
        if candidate is None:
            raise ValueError("该候选已失效，请刷新全图分析。")
        if candidate["state"] not in {"available", "excluded"}:
            raise ValueError("已入账或正在核验的候选不能修改排除状态。")
        state = "excluded" if excluded else "available"
        if candidate["state"] == state:
            return
        candidate["state"] = state
        self.candidatesChanged.emit(self.candidates)
        self.candidateStateChanged.emit()

    def cancel_candidate_submissions(self):
        if self._confirmation and self._confirmation["kind"] == "automatic":
            self._cancel_confirmation()
        while self._candidate_queue:
            self._release_candidate_request(self._candidate_queue.popleft())
        self._refresh_pending = None
        self.candidatesChanged.emit(self.candidates)
        self.candidateStateChanged.emit()

    def _release_candidate_request(self, request):
        for original in request.get("candidates") or ():
            candidate = next((c for c in self.candidates if c.get("candidate_id") == original.get("candidate_id")), None)
            if candidate is not None and candidate.get("state") == "confirming":
                candidate["state"] = "available"
        self.candidatesChanged.emit(self.candidates)
        self.candidateStateChanged.emit()

    def _advance_candidate_queue(self):
        if self._confirmation is not None or not self._candidate_queue:
            return
        request = self._candidate_queue.popleft()
        if request["context"] is not self._automatic_context:
            self._release_candidate_request(request)
            return
        self._queue_confirmation("automatic", request["intents"], context=request["context"],
                                 candidates=request["candidates"])

    def accept_candidates(self, selections: list[tuple[int, str]], *, confirmed_out_of_range=(), _queue=False):
        if self._confirmation is not None and (not _queue or self._confirmation["kind"] != "automatic"):
            raise ValueError("正在核验上次确认时的视场，请等待结果。")
        context = self._automatic_context
        if not context or context["token"].field_epoch != self.field_epoch or self.state != "ready":
            raise ValueError("候选所属视场已失效。")
        if self.presented is None:
            raise ValueError("正在核验最新视场，请稍后确认。")
        if not self.candidate_overlay_ready:
            raise ValueError("正在核验候选与当前画面对齐，请稍后点击。")
        if not selections:
            raise ValueError("请至少勾选一条候选。")
        # Validate the entire selection before mutating the ledger.
        indices = [index for index, _ in selections]
        if any(type(index) is not int or not 0 <= index < len(self.candidates) for index in indices):
            raise ValueError("候选序号无效。")
        if len(indices) != len(set(indices)):
            raise ValueError("候选不能重复提交。")
        if _queue and self.candidate_pending_count + len(selections) > 16:
            raise ValueError("候选核验队列已满（16 条），请等待完成后继续点击。")
        self._ensure_candidate_ids()
        if any(self.candidates[index]["state"] != "available" for index in indices):
            raise ValueError("该候选已入账、正在核验或已排除，不能重复提交。")
        confirmed = set(confirmed_out_of_range)
        if any(type(index) is not int or index not in indices for index in confirmed):
            raise ValueError("越限确认必须对应本次勾选的候选。")
        intents = []
        for index, material in selections:
            result = deepcopy(self.candidates[index])
            result["diameter_um"] = float(result["width_px"]) * context["calibration"].x_um
            if not math.isfinite(result["diameter_um"]) or result["diameter_um"] <= 0:
                raise ValueError("候选直径无效，本次结果未计入。")
            result["_candidate_point_selection"] = _queue
            self._validate_point(result["start"], context["token"].frame)
            self._validate_point(result["end"], context["token"].frame)
            revision = context.get("diagnostics", {}).get("revision", 1)
            result["revision"] = f"cu-automatic-{context['algorithm']}-v{revision}"
            intent = MeasurementIntent(uuid.uuid4().hex, context["token"], tuple(result["start"]),
                tuple(result["end"]), context["calibration"], context["operator_id"],
                context["direction"], material, "automatic-" + context["algorithm"],
                threading.Event(), context["lease"], time.perf_counter())
            if not self._valid(intent):
                raise ValueError("存在未赋类、越限或已失效候选，请先处理。")
            if self._out_of_range(intent, result):
                if index not in confirmed:
                    raise ValueError("存在尚未确认保留的越限候选；请改类、排除或明确确认保留。")
                intent.range_confirmation = self._range_signature(intent, result)
            intents.append((intent, result))
        self.candidate_interaction_started()
        originals = [deepcopy(self.candidates[index]) for index, _ in selections]
        for index in indices:
            self.candidates[index]["state"] = "confirming"
        self.candidatesChanged.emit(self.candidates)
        request = {"intents": intents, "context": context, "candidates": originals}
        self._candidate_queue.append(request)
        self._advance_candidate_queue()
        self.candidateStateChanged.emit()

    def _commit_candidates(self, context, intents, original_candidates):
        # Recheck current material bounds after the asynchronous motion gate;
        # endpoint geometry and selected classes are snapshots of the action.
        for intent, result in intents:
            self._require_current_presented_verified(intent)
            self._validate_range_confirmation(intent, result)
            if result.get("_candidate_point_selection"):
                index = next((i for i, candidate in enumerate(self.candidates)
                    if candidate.get("candidate_id") == result.get("candidate_id")), None)
                if index is None or self.possible_duplicate_candidates([index]):
                    raise ValueError("该候选可能与刚入账的记录重复，请在审核页核对后确认。")
        if not self.candidate_overlay_ready:
            raise ValueError("候选与最新画面尚未完成对齐核验，本次结果未入账。")
        evidence_entries = [self._reserve_evidence(intent, result) for intent, result in intents]
        for intent, _ in intents:
            self._measurement_number += 1
            intent.measurement_number = self._measurement_number
        entries = [self._entry(intent, result, evidence) for (intent, result), evidence in zip(intents, evidence_entries)]
        for entry, original in zip(entries, original_candidates):
            entry["provenance"]["candidate_id"] = original["candidate_id"]
            entry["provenance"]["candidate_ordinal"] = original["ordinal"]
            entry["provenance"]["candidate_batch_id"] = context.get("batch_id")
            entry["provenance"]["analysis_latency_ms"] = context.get("latency_ms")
        records = self.session.add_measurements_batch(entries, provenance={
            "source_frame": context["token"].frame.frame_id, "algorithm": context["algorithm"],
            "field_id": self.field_id, "diagnostics": context.get("diagnostics", {}),
            "original_candidates": [{key: value for key, value in candidate.items()
                if key not in {"candidate_id", "ordinal", "state", "record_id"}}
                for candidate in original_candidates]})
        self._ensure_field(intents[0][0], records[0].created_at)
        self.current_records.extend(record.id for record in records)
        for record, original, evidence in zip(records, original_candidates, evidence_entries):
            self.candidate_metrics["click_to_accept_ms"].append(record.provenance["latency_ms"])
            candidate = next((c for c in self.candidates if c.get("candidate_id") == original["candidate_id"]), None)
            if candidate is not None:
                candidate.update(state="accepted", record_id=record.id)
            if evidence.get("state") == "pending":
                self._evidence_records.setdefault(evidence.get("evidence_id", context["token"].frame.frame_id), []).append(record.id)
        # Domain transaction completes before any synchronous UI callback runs.
        self.recordsAccepted.emit(records)
        self.changed.emit()
        self.candidatesChanged.emit(self.candidates)
        self.candidateStateChanged.emit()

    def possible_duplicate_candidates(self, indices):
        """Warn about nearby geometry; never silently merge or reject a fiber."""
        cells = {}
        for record_id in self.current_records:
            record = self.session.measurements.get(record_id)
            if record is None or record.endpoints is None:
                continue
            a, b = record.endpoints
            center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            cell = (math.floor(center[0] / 8), math.floor(center[1] / 8))
            cells.setdefault(cell, []).append(record)
        duplicates = []
        context = self._automatic_context
        if not context:
            return duplicates
        for index in indices:
            candidate = self.candidates[index]
            a, b = candidate["start"], candidate["end"]
            center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            cell = (math.floor(center[0] / 8), math.floor(center[1] / 8))
            width = candidate["width_px"] * context["calibration"].x_um
            nearby = (record for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for record in cells.get((cell[0] + dx, cell[1] + dy), ()))
            for record in nearby:
                c, d = record.endpoints
                distance = min(max(math.dist(a, c), math.dist(b, d)), max(math.dist(a, d), math.dist(b, c)))
                if distance <= 3 and abs(width - record.diameter_um) <= max(.5, width * .05):
                    duplicates.append(index)
                    break
        return duplicates

    @property
    def busy(self):
        return bool(self._jobs or self.pending_classification or self._confirmation or self._candidate_queue
                    or self._automatic_context and (not self._automatic_context.get("completed")
                        or any(c.get("state", "available") in {"available", "confirming"} for c in self.candidates))
                    or self.evidence.pending)

    def close(self, *, record_end=True):
        if self._closed:
            return
        self._timer.stop()
        self._performance.stop()
        self.end_field("试验工作区已关闭", record_end=record_end)
        self._closed = True
        self._manual_pool.shutdown(wait=True, cancel_futures=False)
        for intent, _ in self._jobs:
            self.budget.release(intent.lease)
        self._jobs.clear()
        # The at-most-one queued motion job owns a frame lease. Let its bounded
        # reduction run so its finally releases that lease even during close.
        self._motion_pool.shutdown(wait=True, cancel_futures=False)
        self._alignment.close()
        if self._prepare_pool is not None:
            self._prepare_pool.shutdown(wait=True, cancel_futures=False)
        if self._prepare_future is not None:
            self._release_motion_item(self._prepare_future[2])
            self._prepare_future = None
        self._release_motion_item(self._motion_item)
        self._motion_item = None
        self._motion_future = None
        self._preparer.previous = None
        self._motion.reset()
        if self._automatic:
            self._automatic.close()
        for entry in self.evidence.close():
            for record_id in self._evidence_records.pop(entry.get("evidence_id", entry["frame_id"]), []):
                self.session.set_evidence(record_id, entry)
        self.presented = self.latest_frame = None
        self._release_frame_reference("_latest_frame_lease")
        self._release_frame_reference("_presented_frame_lease")
        self._release_frame_reference("_presentation_lease")
