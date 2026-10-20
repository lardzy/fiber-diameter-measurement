"""Fixed-reference candidate alignment, independent of business-field motion.

One worker and one latest-frame slot; every raw reference and scratch allocation
is admitted by the live frame budget. Workers never publish GUI state.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .frames import FrameBudgetExceeded


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    valid: bool
    reason: str
    shift: tuple[float, float] = (0., 0.)
    response: float = 0.
    correlation: float = 0.
    gradient_ratio: float = 0.
    elapsed_ms: float = 0.


def _gray(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.


def _energy(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(dx * dx + dy * dy))


class CandidateReferenceCache:
    """Prepared once by the serial worker, charged to its retained scratch.

    The frame owner remains under the monitor/reference and in-flight leases.
    Every generation gets a separate object, so reset cannot mutate preparation
    still used by an older in-flight comparison.
    """
    def __init__(self):
        self.reference = None
        self.normalized = None
        self.std = self.energy = 0.
        self.window = None

    def prepare(self, reference):
        if self.reference is reference:
            return
        a = _gray(reference)
        self.reference = reference
        self.std = float(a.std())
        if self.std < .01:
            return
        self.normalized = (a - float(a.mean())) / self.std
        self.energy = _energy(self.normalized)
        self.window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        self.normalized.setflags(write=False)
        self.window.setflags(write=False)


def compare_candidate_frames(reference, current, *, prepared=None):
    """Conservative native-resolution check; never translate candidate geometry."""
    started = time.perf_counter()
    if reference.shape != current.shape or min(reference.shape[:2]) < 16:
        return AlignmentResult(False, "frame_shape")
    prepared = CandidateReferenceCache() if prepared is None else prepared
    prepared.prepare(reference)
    b = _gray(current)
    sa, sb = prepared.std, float(b.std())
    if min(sa, sb) < .01:
        return AlignmentResult(False, "low_texture")
    a = prepared.normalized
    b = (b - float(b.mean())) / sb
    correlation = float(np.mean(a * b))
    ea, eb = prepared.energy, _energy(b)
    ratio = eb / ea if ea > 1e-8 else float("nan")
    # phaseCorrelate multiplies input by the window in place. Never pass cached
    # normalized pixels directly: repeated frames must not repeatedly window it.
    shift, response = cv2.phaseCorrelate(a.copy(), b, prepared.window)
    finite = all(math.isfinite(v) for v in (*shift, response, correlation, ratio))
    valid = finite and math.hypot(*shift) <= 1. and response >= .6 and correlation >= .95 and .5 <= ratio <= 2.
    return AlignmentResult(valid, "aligned" if valid else "alignment_lost", tuple(shift),
                           float(response), correlation, ratio, (time.perf_counter() - started) * 1000)


class CandidateAlignmentMonitor:
    """The caller owns source identity and consumes only the current generation."""
    def __init__(self, budget, wake, *, worker=None):
        from .candidate_alignment_worker import AlignmentProcessWorker
        self.budget, self.wake = budget, wake
        self.worker = AlignmentProcessWorker() if worker is None else worker
        self._failed_retirements = []
        self._reference_used = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-candidate-alignment")
        self.reference = None
        self.lease = None
        self.reference_scratch = None
        self.preparation = None
        self.generation = 0
        self.future = None
        self.last_submitted = None
        self.result = None
        self.completed_frame = None
        self.completed_submitted_at = None
        self.completed_ready_at = None
        self._completed_lease = None
        # Scalar identities only: preserve the previous verified display while
        # a newer verified frame is waiting for Qt to paint, without treating
        # skipped intermediate frames as verified by a later result.
        self._results = deque(maxlen=8)
        self.closed = False

    def reset(self, frame=None):
        self.generation += 1
        self._release_completed()
        if self.lease is not None:
            self.budget.release(self.lease)
        if self.reference_scratch is not None:
            if self._reference_used:
                self._retire_preparation(self.reference_scratch)
            else:
                self.budget.release_scratch(self.reference_scratch)
        self._reference_used = False
        self.lease = None
        self.reference_scratch = None
        self.preparation = None
        self.reference = None
        self.last_submitted = None
        self.result = None
        self._results.clear()
        if frame is not None:
            # Keep preparation charged between jobs. The serial group's 64 B/px
            # bound includes cached arrays plus comparison/FFT workspaces.
            self.lease, self.reference_scratch = self.budget.acquire_with_scratch(
                frame, group="candidate-alignment", scratch_bytes=frame.size[0] * frame.size[1] * 64)
            self.reference = frame
            # Preparation lives only in the isolated process. The retained
            # reservation also covers its two shared input buffers.

    def _retire_preparation(self, ticket):
        # Queue cleanup after the work that may create this generation's cache.
        # Do not coalesce tickets into a currently running cleanup: a newer
        # generation's work may already be queued *behind* that cleanup.
        # Unused resets enqueue nothing; one outstanding comparison bounds the
        # executor to one active cleanup plus the next work/cleanup pair.
        self.pool.submit(self._drop_retired_preparation, ticket)

    def _drop_retired_preparation(self, ticket):
        try:
            self.worker.invalidate()
        except Exception:
            # An OS-level failure to reap our child must retain its reservation
            # and stop new submissions, never claim that its pixels were freed.
            self._failed_retirements.append(ticket)
        else:
            self.budget.release_scratch(ticket)

    def _release_completed(self):
        if self._completed_lease is not None:
            self.budget.release(self._completed_lease)
        self._completed_lease = None
        self.completed_frame = None
        self.completed_submitted_at = None
        self.completed_ready_at = None

    def _work(self, reference, current, preparation, leases, scratch):
        try:
            return self.worker.compare(reference.pixels, current.pixels)
        finally:
            for lease in leases:
                self.budget.release(lease)
            self.budget.release_scratch(scratch)

    def _run_work(self, reference, current, preparation, leases, scratch):
        # Mark readiness on the parent worker thread, before Qt consumes it.
        # Delay injection belongs in _work so it cannot bypass this timestamp.
        try:
            result = self._work(reference, current, preparation, leases, scratch)
        except Exception:
            result = AlignmentResult(False, "alignment_error")
        return result, time.perf_counter()

    def poll(self, frame):
        completed = None
        if self.future is not None and self.future[0].done():
            future, generation, identity, sequence, frame_id, current, lease, submitted_at = self.future
            self.future = None
            try:
                result, ready_at = future.result()
            except Exception:
                result, ready_at = AlignmentResult(False, "alignment_error"), time.perf_counter()
            if generation == self.generation:
                self._release_completed()
                # Transfer the worker's current-frame lease to the GUI result.
                # A valid completion may be published atomically even if a
                # newer input has arrived; it is never relabelled as that input.
                self.completed_frame = current
                self._completed_lease = lease
                self.completed_submitted_at = submitted_at
                self.completed_ready_at = ready_at
                self.result = completed = (identity, sequence, frame_id, result)
                self._results.append(completed)
            else:
                self.budget.release(lease)
        if self.closed or self._failed_retirements or frame is None or self.reference is None or self.future is not None:
            return completed
        identity = (frame.source_id, frame.generation, frame.size)
        reference = self.reference
        if identity != (reference.source_id, reference.generation, reference.size):
            return completed
        if frame.frame_id == self.last_submitted:
            return completed
        first = second = scratch = None
        try:
            first = self.budget.acquire(reference)
            second, scratch = self.budget.acquire_with_scratch(frame, group="candidate-alignment",
                scratch_bytes=frame.size[0] * frame.size[1] * 64)
            submitted_at = time.perf_counter()
            future = self.pool.submit(self._run_work, reference, frame, self.preparation, (first,), scratch)
            self._reference_used = True
        except FrameBudgetExceeded:
            if first is not None:
                self.budget.release(first)
            return completed
        except Exception:
            for lease in (first, second):
                if lease is not None:
                    self.budget.release(lease)
            if scratch is not None:
                self.budget.release_scratch(scratch)
            raise
        self.future = (future, self.generation, identity, frame.sequence, frame.frame_id,
                       frame, second, submitted_at)
        self.last_submitted = frame.frame_id
        future.add_done_callback(self.wake)
        return completed

    def result_for(self, frame):
        identity = (frame.source_id, frame.generation, frame.size)
        return next((item for item in reversed(self._results)
                     if item[0] == identity and item[2] == frame.frame_id), None)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.worker.request_stop()
        self.reset()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.worker.close()
        for ticket in self._failed_retirements:
            self.budget.release_scratch(ticket)
        self._failed_retirements.clear()
        if self.future is not None:
            self.budget.release(self.future[6])
        self.future = None
