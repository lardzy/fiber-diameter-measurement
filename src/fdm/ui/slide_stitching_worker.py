"""A single bounded process shared by capture and existing-slide repair."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from queue import Empty, Full

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass(frozen=True, slots=True)
class StitchJob:
    key: str
    source: str
    result: str
    checkpoint: str
    final: bool = True
    document_id: str = ""
    publish_target: str = ""
    open_when_ready: bool = False


def _run_job(job: StitchJob, stop, events) -> None:
    # Top-level spawn target, also usable in a frozen, console-free Windows app.
    import cv2
    from fdm.services.slide_registration import register_slide
    from fdm.services.slide_layout import save_layout
    cv2.setNumThreads(1)
    def report(done, total):
        try:
            events.put_nowait((done, total))
        except Full:
            pass
    try:
        layout = register_slide(job.source, cancelled=stop.is_set, progress=report,
            checkpoint=job.checkpoint, final=job.final)
        if stop.is_set():
            return
        if job.final:
            # Archive each immutable version; the named result is only a latest pointer.
            destination = Path(job.result)
            save_layout(destination.parent / "versions" / f"{layout.layout_id}.fdmstitch", layout)
            save_layout(job.result, layout)
    except InterruptedError:
        return
    except Exception as exc:
        from fdm.atomic_io import atomic_write_json
        atomic_write_json(Path(job.result + ".error"), {"error": str(exc)})
        raise SystemExit(1) from None


class SlideStitchingController(QObject):
    progress = Signal(object, int, int)
    finished = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._context = multiprocessing.get_context("spawn")
        self._pending = OrderedDict()
        self._paused = set()
        self._process = None
        self._active = None
        self._stop = None
        self._events = None
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def submit(self, job: StitchJob):
        if self._closed or job.key in self._paused:
            return
        if self._active is not None and self._active.key == job.key and self._active.final and not self._stop.is_set():
            return
        if job.key not in self._pending and len(self._pending) >= 32:
            self.failed.emit(job, "待检查切片已达到 32 个；请稍后再次启动检查")
            return
        if self._active is not None and self._active.key == job.key and job.final and not self._active.final:
            self._stop.set()
        existing = self._pending.get(job.key)
        if existing is None or not existing.final or job.final:
            self._pending[job.key] = job
        self._timer.start()
        self._launch_next()

    def _launch_next(self):
        if self._process is not None or not self._pending or self._closed:
            return
        _, self._active = self._pending.popitem(last=False)
        self._stop = self._context.Event()
        self._events = self._context.Queue(maxsize=8)
        self._process = self._context.Process(target=_run_job, args=(self._active, self._stop, self._events), daemon=True)
        try:
            self._process.start()
        except Exception as exc:
            job = self._active
            self._release()
            self.failed.emit(job, str(exc))

    def cancel(self, key=None):
        if key is None:
            self._pending.clear()
        else:
            self._pending.pop(key, None)
        if self._active is not None and (key is None or self._active.key == key):
            self._stop.set()

    def pause(self, keys=()):
        self._paused.update(keys)
        self._paused.update(self._pending)
        if self._active is not None:
            self._paused.add(self._active.key)
        self.cancel()

    def resume(self, key):
        self._paused.discard(key)

    def _release(self):
        if self._events is not None:
            self._events.cancel_join_thread()
            self._events.close()
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=0)
            if not self._process.is_alive():
                self._process.close()
        self._process = self._active = self._events = self._stop = None

    def _poll(self):
        if self._process is None:
            self._launch_next()
            if self._process is None and not self._pending:
                self._timer.stop()
            return
        latest = None
        while True:
            try:
                latest = self._events.get_nowait()
            except Empty:
                break
        if latest is not None:
            self.progress.emit(self._active, *latest)
        if self._process.is_alive():
            return
        job, code, cancelled = self._active, self._process.exitcode, self._stop.is_set()
        self._release()
        if not cancelled and job.final:
            if code == 0:
                try:
                    from fdm.services.slide_layout import load_layout
                    layout = load_layout(job.result)
                except Exception as exc:
                    self.failed.emit(job, str(exc))
                else:
                    self.finished.emit(job, layout)
            else:
                message = f"后台拼接进程退出 ({code})；原始切片不受影响"
                try:
                    import json
                    message = json.loads(Path(job.result + ".error").read_text(encoding="utf-8"))["error"]
                except (OSError, ValueError, KeyError):
                    pass
                self.failed.emit(job, message)
        self._launch_next()

    def shutdown(self):
        self._closed = True
        self._timer.stop()
        self.cancel()
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=.3)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=.5)
        self._release()
