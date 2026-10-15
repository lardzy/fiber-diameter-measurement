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
    candidatesChanged = Signal(object)
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
        self.state = "unavailable"
        self.draft: tuple[PresentedFrameToken, tuple[float, float]] | None = None
        self.current_records: list[str] = []
        # Identity/state only: never retain a completed intent's full frame.
        self._classification_target = None
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
            self.unavailable("图像来源或尺寸已变化，旧标定已清除；请等待新画面并重新标定。")
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
            self.resultRejected.emit(str(exc))
            raise

    def mark_presented(self, token: PresentedFrameToken):
        if token.field_epoch == self.field_epoch and self.latest_frame is not None:
            if self.presented is None or self.presented.frame is not token.frame:
                self._release_frame_reference("_presented_frame_lease")
                self._presented_frame_lease = self.budget.acquire(token.frame, preview=True)
            self.presented = token
            self._release_frame_reference("_presentation_lease")

    def unavailable(self, reason: str):
        if self.state != "unavailable":
            self.end_field(reason)
        self.latest_frame = self.presented = None
        self._release_frame_reference("_latest_frame_lease")
        self._release_frame_reference("_presented_frame_lease")
        self._release_frame_reference("_presentation_lease")
        self.calibration = None
        self.state = "unavailable"
        self.statusChanged.emit(reason)

    def end_field(self, reason="视场已变化，请重新测量"):
        if self.field_id in self.session.fields:
            self.session.update_field(self.field_id, ended_at=datetime.now(timezone.utc).isoformat(), end_reason=reason)
        self.field_epoch += 1
        self.field_id = uuid.uuid4().hex
        self.draft = None
        self._release_frame_reference("_draft_lease")
        self.current_records.clear()
        self._classification_target = None
        self._measurement_number = 0
        for intent, _ in self._jobs:
            intent.cancel.set()
        for intent, _ in self.pending_classification:
            self.budget.release(intent.lease)
        self.pending_classification.clear()
        self._cancel_confirmation()
        self.candidates = []
        if self._automatic_context:
            self._automatic.cancel()
            self.budget.release(self._automatic_context["lease"])
            self._automatic_context = None
        self._motion_reset = True
        self._motion_sequence = -1
        self._last_prepare_sequence = -1
        self._stable_count = 0
        self.state = "unavailable" if self.latest_frame is None else "stabilizing" if self.latest_frame.live else "ready"
        self.fieldChanged.emit(self.field_epoch, reason)
        self.candidatesChanged.emit([])
        self.statusChanged.emit(reason)
        self.changed.emit()

    def start(self, point: tuple[float, float]):
        if self._closed:
            raise RuntimeError("试验工作区已关闭。")
        if (self.state != "ready" or self.presented is None
                or self.presented.field_epoch != self.field_epoch):
            raise ValueError("请等待图像来源和视场稳定。")
        if self.calibration is None:
            raise ValueError("请先设置当前图像来源的标定。")
        self._validate_point(point, self.presented.frame)
        lease = self._acquire_intent_frame(self.presented.frame)
        self._release_frame_reference("_draft_lease")
        self._draft_lease = lease
        self.draft = (self.presented, point)
        self._measurement_number += 1
        self._classification_target = {"id": uuid.uuid4().hex, "number": self._measurement_number, "field_epoch": self.field_epoch,
            "operator_id": self.session.active_operator_id, "direction": self.session.active_direction,
            "state": "draft", "record_id": None}

    def cancel_draft(self):
        if self.draft is not None and self._classification_target is not None:
            self._classification_target["state"] = "cancelled"
        self.draft = None
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
        if token.field_epoch != self.field_epoch or self.state != "ready":
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
        self._release_frame_reference("_draft_lease")
        identity = self._classification_target["id"] if self._classification_target else uuid.uuid4().hex
        intent = MeasurementIntent(identity, token, start, tuple(point), self.calibration,
            self.session.active_operator_id, self.session.active_direction, material_id,
            method, threading.Event(), lease, time.perf_counter())
        if self._classification_target is not None:
            self._classification_target["state"] = "pending"
            intent.measurement_number = self._classification_target["number"]
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
        return (not intent.cancel.is_set() and intent.token.field_epoch == self.field_epoch
                and intent.calibration == self.calibration
                and intent.operator_id == self.session.active_operator_id
                and intent.direction == self.session.active_direction)

    def _set_classification_state(self, intent, state, record_id=None):
        target = self._classification_target
        if target is not None and target["id"] == intent.id:
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
        self._presentation_lease = self.budget.replace(self._presentation_lease, item.frame, preview=True)
        self.frameReady.emit(item.frame)

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
        deadline = time.perf_counter() + 0.004
        while self._jobs and self._jobs[0][1].done() and time.perf_counter() < deadline:
            intent, future = self._jobs[0]
            if self._valid(intent) and self.presented and self.presented.frame.live:
                # Every publication must cover the currently displayed frame.
                # Advance the target, but never restart the original timeout.
                self._advance_submission_barrier(intent, self.presented.frame.sequence)
                elapsed = time.perf_counter() - intent.submission_barrier_at
                if elapsed > 2:
                    intent.cancel.set()
                    self.resultRejected.emit("视场核验超过两秒，本次结果未入账；请等待视场稳定后重试。")
                elif self._motion_sequence < intent.submission_barrier_sequence:
                    if elapsed > .5 and self.state == "ready":
                        self.state = "stabilizing"
                        self.statusChanged.emit("视场核验较慢，正在暂停新测量并等待当前结果。")
                    break
            self._jobs.popleft()
            keep_lease = False
            try:
                if not self._valid(intent):
                    continue
                result = future.result()
                if result.get("status") not in {"ok", "success", "valid"} or not (
                        math.isfinite(float(result.get("diameter_um", 0))) and result["diameter_um"] > 0):
                    self.resultRejected.emit(str(result.get("reason") or "未找到有效的两侧边缘"))
                    continue
                if self._out_of_range(intent, result):
                    self.pending_classification.append((intent, result))
                    keep_lease = True
                    self.statusChanged.emit("测量超出类别范围，请在待确认项中保留、改类或放弃。")
                    self.changed.emit()
                else:
                    self._accept(intent, result)
            except Exception as exc:
                self.resultRejected.emit(str(exc))
            finally:
                if not keep_lease:
                    target = self._classification_target
                    if target is not None and target["id"] == intent.id and target["state"] == "pending":
                        self._set_classification_state(intent, "failed")
                    self.budget.release(intent.lease)
        self._poll_confirmation()
        for entry in self.evidence.poll():
            for record_id in self._evidence_records.pop(entry["frame_id"], []):
                self.session.set_evidence(record_id, dict(entry))
            self.changed.emit()
        self._poll_automatic()

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
        evidence = dict(self.evidence.reserve(intent.token.frame)) if self.session.save_images else {"state": "disabled"}
        record = self.session.add_measurement(**self._entry(intent, result, evidence))
        self._ensure_field(intent, record.created_at)
        self.current_records.append(record.id)
        self._set_classification_state(intent, "accepted", record.id)
        if evidence.get("state") == "pending":
            self._evidence_records.setdefault(intent.token.frame.frame_id, []).append(record.id)
        self.recordAccepted.emit(record)
        self.changed.emit()
        return record

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

    def _poll_confirmation(self):
        request = self._confirmation
        if request is None:
            return
        valid = all(self._valid(intent) for intent, _ in request["intents"])
        if request["kind"] == "automatic":
            valid = valid and self._automatic_context is request["context"]
        if not valid or self.presented is None:
            self._cancel_confirmation()
            self.resultRejected.emit("视场已变化，本次确认未入账。")
            self.changed.emit()
            return
        sequence = self.presented.frame.sequence if self.presented.frame.live else None
        request["sequence"] = sequence
        for intent, _ in request["intents"]:
            self._advance_submission_barrier(intent, sequence)
        elapsed = time.perf_counter() - request["started"]
        if sequence is not None and elapsed > 2:
            self._cancel_confirmation()
            self.resultRejected.emit("视场核验超过两秒，本次确认未入账；请等待视场稳定后重试。")
            self.changed.emit()
            return
        if sequence is not None and self._motion_sequence < sequence:
            if elapsed > .5 and self.state == "ready":
                self.state = "stabilizing"
                self.statusChanged.emit("视场核验较慢，正在暂停新测量并等待本次确认。")
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
            self.resultRejected.emit(str(exc))
        finally:
            if request["kind"] == "manual":
                for intent, _ in request["intents"]:
                    self.budget.release(intent.lease)
            self.changed.emit()

    def analyze(self, algorithm="traditional"):
        if self._closed:
            raise RuntimeError("试验工作区已关闭。")
        if self.state != "ready" or self.presented is None or self.calibration is None:
            raise ValueError("请等待视场稳定并完成标定。")
        if not self.calibration.isotropic:
            raise ValueError("全图测径需要等比例标定。")
        if self._automatic_context is not None:
            raise ValueError("请先确认、放弃或取消上次全图分析。")
        from fdm.services.cu_automatic_worker import CuAutomaticWorker
        if self._automatic is None:
            self._automatic = CuAutomaticWorker()
        token = self.presented
        lease = self._acquire_intent_frame(token.frame)
        try:
            task_id = self._automatic.submit(token.frame.pixels, algorithm=algorithm)
        except Exception:
            self.budget.release(lease)
            raise
        self._automatic_context = {"task_id": task_id, "token": token, "lease": lease,
            "calibration": self.calibration, "algorithm": algorithm,
            "operator_id": self.session.active_operator_id, "direction": self.session.active_direction}
        self.statusChanged.emit("正在分析当前视场；实时画面继续更新。")

    def _poll_automatic(self):
        if self._automatic is None:
            return
        for response in self._automatic.poll():
            context = self._automatic_context
            if not context or response.get("task_id") != context["task_id"]:
                continue
            result = response.get("result") or {}
            if context["token"].field_epoch != self.field_epoch:
                self.discard_candidates()
            elif result.get("status") in {"ok", "success", "empty"}:
                context["diagnostics"] = result.get("diagnostics", {})
                self.candidates = list(result.get("candidates", []))
                self.candidatesChanged.emit(self.candidates)
                self.statusChanged.emit(f"检测到 {len(self.candidates)} 条候选，请赋类后确认。" if self.candidates else "未检测到符合当前算法规则的纤维。")
                if not self.candidates:
                    self.discard_candidates()
            else:
                self.resultRejected.emit(str(response.get("error") or result.get("reason") or "全图分析失败"))
                self.discard_candidates()

    def discard_candidates(self):
        if self._confirmation and self._confirmation["kind"] == "automatic":
            self._cancel_confirmation()
        if self._automatic_context:
            self._automatic.cancel()
            self.budget.release(self._automatic_context["lease"])
        self._automatic_context = None
        self.candidates = []
        self.candidatesChanged.emit([])

    def accept_candidates(self, selections: list[tuple[int, str]], *, confirmed_out_of_range=()):
        if self._confirmation is not None:
            raise ValueError("正在核验上次确认时的视场，请等待结果。")
        context = self._automatic_context
        if not context or context["token"].field_epoch != self.field_epoch or self.state != "ready":
            raise ValueError("候选所属视场已失效。")
        if self.presented is None:
            raise ValueError("正在核验最新视场，请稍后确认。")
        if not selections:
            raise ValueError("请至少勾选一条候选。")
        # Validate the entire selection before mutating the ledger.
        indices = [index for index, _ in selections]
        if any(type(index) is not int or not 0 <= index < len(self.candidates) for index in indices):
            raise ValueError("候选序号无效。")
        if len(indices) != len(set(indices)):
            raise ValueError("候选不能重复提交。")
        confirmed = set(confirmed_out_of_range)
        if any(type(index) is not int or index not in indices for index in confirmed):
            raise ValueError("越限确认必须对应本次勾选的候选。")
        intents = []
        for index, material in selections:
            result = deepcopy(self.candidates[index])
            result["diameter_um"] = float(result["width_px"]) * context["calibration"].x_um
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
        self._queue_confirmation("automatic", intents, context=context,
            candidates=[deepcopy(self.candidates[index]) for index, _ in selections])

    def _commit_candidates(self, context, intents, original_candidates):
        # Recheck current material bounds after the asynchronous motion gate;
        # endpoint geometry and selected classes are snapshots of the action.
        for intent, result in intents:
            self._require_current_presented_verified(intent)
            self._validate_range_confirmation(intent, result)
        evidence = dict(self.evidence.reserve(context["token"].frame)) if self.session.save_images else {"state": "disabled"}
        for intent, _ in intents:
            self._measurement_number += 1
            intent.measurement_number = self._measurement_number
        entries = [self._entry(intent, result, evidence) for intent, result in intents]
        records = self.session.add_measurements_batch(entries, provenance={
            "source_frame": context["token"].frame.frame_id, "algorithm": context["algorithm"],
            "field_id": self.field_id, "diagnostics": context.get("diagnostics", {}),
            "original_candidates": original_candidates})
        self._ensure_field(intents[0][0], records[0].created_at)
        self.current_records.extend(record.id for record in records)
        target = self._classification_target
        if target is None or target["state"] not in {"draft", "pending"}:
            last = intents[-1][0]
            self._classification_target = {"id": last.id, "number": last.measurement_number,
                "field_epoch": self.field_epoch, "operator_id": last.operator_id,
                "direction": last.direction, "state": "accepted", "record_id": records[-1].id}
        if evidence.get("state") == "pending":
            self._evidence_records.setdefault(context["token"].frame.frame_id, []).extend(record.id for record in records)
        # Domain transaction completes before any synchronous UI callback runs.
        self.recordsAccepted.emit(records)
        self.changed.emit()
        self.discard_candidates()

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
        return bool(self._jobs or self.pending_classification or self._confirmation or self._automatic_context or self.evidence.pending)

    def close(self):
        if self._closed:
            return
        self._timer.stop()
        self.end_field("试验工作区已关闭")
        self._closed = True
        self._manual_pool.shutdown(wait=True, cancel_futures=False)
        for intent, _ in self._jobs:
            self.budget.release(intent.lease)
        self._jobs.clear()
        # The at-most-one queued motion job owns a frame lease. Let its bounded
        # reduction run so its finally releases that lease even during close.
        self._motion_pool.shutdown(wait=True, cancel_futures=False)
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
            for record_id in self._evidence_records.pop(entry["frame_id"], []):
                self.session.set_evidence(record_id, entry)
        self.presented = self.latest_frame = None
        self._release_frame_reference("_latest_frame_lease")
        self._release_frame_reference("_presented_frame_lease")
        self._release_frame_reference("_presentation_lease")
