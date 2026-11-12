"""Serial camera I/O with Qt-thread completion and cancellable request identity.

Cancellation retires intent, not an uninterruptible driver call. The running
operation keeps the device lease until its completion/cleanup; another operation
cannot run concurrently. QObject/SDK window work can explicitly remain on Qt.
"""
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import perf_counter

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot


@dataclass
class DeviceJob:
    epoch: int
    stage: str
    function: object
    complete: object
    worker: bool
    always: bool = False
    orphan_cleanup: object = None
    ready: object = field(default_factory=Event)
    claim_lock: object = field(default_factory=Lock)
    claimed: bool = False

    def cleanup_if_unclaimed(self):
        self.ready.wait()
        with self.claim_lock:
            if self.claimed:
                return
            self.claimed = True
        if self.orphan_cleanup is not None:
            try:
                self.orphan_cleanup()
            except Exception:
                # A dead GUI has no receiver. Backend stop failure is not an
                # authorization to open a second device or kill its driver.
                pass


class CaptureDeviceJobs(QObject):
    finished = Signal(object)
    timing = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.epoch = 0
        self._queue = deque()
        self._running = None
        self._owner_destroyed = Event()
        self.finished.connect(self._complete, Qt.ConnectionType.QueuedConnection)
        self.destroyed.connect(lambda *_: self._retire_owner())

    def _retire_owner(self, *_):
        self._owner_destroyed.set()
        job = self._running
        releases = [pending for pending in self._queue if pending.always and pending.worker]
        self._queue.clear()
        def cleanup():
            if job is not None and job.worker:
                job.cleanup_if_unclaimed()
            # A completed open can already have handed its lease to a queued
            # release when the GUI is destroyed. Preserve that release too,
            # after any still-running call has finished; never start them in
            # parallel merely because their Qt receiver no longer exists.
            for release in releases:
                try:
                    release.function()
                except Exception:
                    pass
        if (job is not None and job.worker) or releases:
            Thread(target=cleanup, name="fdm-camera-orphan-cleanup", daemon=True).start()

    @property
    def busy(self):
        return self._running is not None or bool(self._queue)

    def is_running(self, job):
        return self._running is job

    def cancel(self):
        self.epoch += 1
        self._queue = deque(job for job in self._queue if job.always)
        return self.epoch

    def submit(self, stage, function, complete, *, worker=True, always=False,
               first=False, orphan_cleanup=None):
        job = DeviceJob(self.epoch, stage, function, complete, worker, always, orphan_cleanup)
        (self._queue.appendleft if first else self._queue.append)(job)
        QTimer.singleShot(0, self, self._pump)
        return job

    def _pump(self):
        if self._running is not None:
            return
        while self._queue:
            job = self._queue.popleft()
            if job.always or job.epoch == self.epoch:
                break
        else:
            return
        self._running = job

        def run():
            started = perf_counter()
            try:
                value, error = job.function(), None
            except Exception as exc:
                value, error = None, exc
            payload = (job, value, error, (perf_counter() - started) * 1000)
            job.ready.set()
            if job.worker:
                if self._owner_destroyed.is_set():
                    job.cleanup_if_unclaimed()
                    return
                try:
                    self.finished.emit(payload)
                except RuntimeError:
                    # The GUI owner may be destroyed while a native open call
                    # is still returning. Only this job's backend is released.
                    job.cleanup_if_unclaimed()
            else:
                self._complete(payload)

        if job.worker:
            Thread(target=run, name="fdm-camera-" + job.stage, daemon=True).start()
        else:
            run()

    @Slot(object)
    def _complete(self, payload):
        job, value, error, elapsed = payload
        if self._running is not job:
            return
        with job.claim_lock:
            if job.claimed:
                return
            job.claimed = True
        self._running = None
        stale = job.epoch != self.epoch and not job.always
        try:
            job.complete(value, error, stale)
        finally:
            if not self._owner_destroyed.is_set():
                self.timing.emit({"stage": job.stage, "elapsed_ms": elapsed, "epoch": job.epoch,
                                  "cancelled": stale, "ok": error is None})
                self._pump()
