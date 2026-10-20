"""Asynchronous full/cropped evidence store; accepted records retain failures."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from .frames import FrameBudget, FrameHandle, acquisition_reservation
from .png_stream import ENCODER_REVISION, png_workspace_bytes, write_png_stream
from .evidence_options import effective_evidence_options, evidence_crop
from .jpeg_evidence import (ENCODER_REVISION as JPEG_ENCODER_REVISION,
                            jpeg_workspace_bytes, write_jpeg_evidence)


@dataclass(slots=True)
class _WriteTask:
    frame: FrameHandle | None
    entry: dict
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    lease: str | None = None
    scratch: str | None = None
    state: str = "prepared"
    lock: object = field(default_factory=threading.Lock)


class EvidenceStore:
    def __init__(self, directory: Path | str | None = None, *, budget: FrameBudget | None = None):
        self.directory = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="fdm-live-evidence-"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budget = budget or FrameBudget()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-evidence")
        self._entries: dict[str, dict] = {}
        self._futures = {}
        self._tasks: dict[str, _WriteTask] = {}
        self._scratch_group = "evidence:" + uuid.uuid4().hex
        self._lock = threading.RLock()
        self._closed = False

    def reserve(self, frame: FrameHandle, *, options: dict | None = None, points=None) -> dict:
        """Reserve before ledger commit; dedup source, rectangle and encoding.

        No options preserves the historical full PNG contract. A crop must have
        original-pixel geometry, never a latest-preview or screen-space point.
        """
        policy = effective_evidence_options(options)
        if policy["mode"] == "disabled":
            return {"state": "disabled"}
        width, height = frame.size
        region = ((0, 0, width, height) if policy["mode"] == "full_png" else
                  evidence_crop(frame.size, () if points is None else points, padding=policy["padding"],
                                minimum_size=policy["minimum_size"]))
        x, y, w, h = region
        codec = "png" if policy["mode"] == "full_png" else "jpeg"
        if codec == "jpeg" and frame.pixels.ndim == 3 and frame.pixels.shape[2] == 4:
            # Reduce directly: an H×W boolean temporary would be unbudgeted
            # pixel memory before admission for a large automatic crop.
            if int(frame.pixels[y:y+h, x:x+w, 3].min()) != 255:
                codec = "png"
        identity = {"frame_id": frame.frame_id, "crop": list(region), "codec": codec,
                    "quality": policy["quality"] if codec == "jpeg" else None,
                    "subsampling": "4:4:4" if codec == "jpeg" else None}
        # Keep legacy full-frame filenames stable, including old test fixtures.
        evidence_id = (hashlib.sha256(frame.frame_id.encode()).hexdigest() if policy["mode"] == "full_png" else
                       hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest())
        name = evidence_id + (".jpg" if codec == "jpeg" else ".png")
        entry = {"state": "pending", "path": str(self.directory / name),
                 "frame_id": frame.frame_id, "sha256": None, "error": None,
                 "evidence_id": evidence_id, "metadata_version": 1,
                 "mode": policy["mode"], "codec": codec, "source_size": [width, height],
                 "crop": list(region), "quality": identity["quality"],
                 "subsampling": identity["subsampling"],
                 "encoder": JPEG_ENCODER_REVISION if codec == "jpeg" else ENCODER_REVISION}
        task = _WriteTask(frame, entry, region)
        channels = 1 if frame.pixels.ndim == 2 else int(frame.pixels.shape[2])
        scratch_bytes = jpeg_workspace_bytes(w, h) if codec == "jpeg" else png_workspace_bytes(w, channels)
        with self._lock:
            if self._closed:
                raise RuntimeError("测量证据存储已关闭。")
            if evidence_id in self._entries:
                return self._entries[evidence_id]
            task.lease, task.scratch = self.budget.acquire_with_scratch(
                frame, group=self._scratch_group, scratch_bytes=scratch_bytes,
                reserve_bytes=acquisition_reservation(frame),
                keep_preview_slots=2 if frame.live else 0)
            # ThreadPoolExecutor enqueues before starting a new thread. If
            # Thread.start raises, that queued work can execute on a later
            # submit even though no Future was returned. Keep the worker behind
            # this lock until registration succeeds, or mark it inert on error.
            with task.lock:
                future = None
                try:
                    future = self._pool.submit(self._write, task)
                    self._entries[evidence_id] = entry
                    self._futures[evidence_id] = future
                    self._tasks[evidence_id] = task
                    task.state = "armed"
                except BaseException:
                    task.state = "aborted"
                    task.frame = None
                    self._entries.pop(evidence_id, None)
                    self._futures.pop(evidence_id, None)
                    self._tasks.pop(evidence_id, None)
                    self._release_task_resources(task)
                    if future is not None:
                        future.cancel()
                    raise
            return entry

    def _release_task_resources(self, task: _WriteTask) -> None:
        """Caller holds task.lock; release is idempotent and never waits."""
        if task.lease is not None:
            self.budget.release(task.lease)
            task.lease = None
        if task.scratch is not None:
            self.budget.release_scratch(task.scratch)
            task.scratch = None

    def _write(self, task: _WriteTask) -> dict | None:
        with task.lock:
            if task.state != "armed":
                return None
            task.state = "running"
            frame, task.frame = task.frame, None
        entry = task.entry
        temporary = None
        try:
            temporary = Path(entry["path"]).with_suffix(".pending")
            with temporary.open("wb", buffering=0) as handle:
                x, y, width, height = task.region
                pixels = frame.pixels[y:y+height, x:x+width]
                encoded = (write_jpeg_evidence(pixels, handle, quality=entry["quality"])
                           if entry["codec"] == "jpeg" else write_png_stream(pixels, handle))
                # The encoder and executor argument no longer own the pixels.
                # Other consumers' leases remain counted until they release.
                pixels = frame = None
                with task.lock:
                    if task.lease is not None:
                        self.budget.release(task.lease)
                        task.lease = None
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, entry["path"])
            return {**entry, "state": "saved", "sha256": encoded.sha256,
                    "size_bytes": encoded.size_bytes}
        except Exception as exc:
            error = str(exc)
            # An error string is persisted, never an exception traceback that
            # could keep the borrowed input or compressor buffers alive.
            exc.__traceback__ = None
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                # Losing write access can also prevent removal. Preserve the
                # accepted record's failure instead of raising from GUI poll.
                error += f"；临时测量证据清理失败：{cleanup_error}"
            return {**entry, "state": "failed", "error": error}
        finally:
            pixels = frame = None
            with task.lock:
                task.state = "done"
                self._release_task_resources(task)

    def poll(self) -> list[dict]:
        completed = []
        with self._lock:
            for evidence_id, future in list(self._futures.items()):
                if future.done():
                    task = self._tasks[evidence_id]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {**task.entry, "state": "failed", "error": str(exc)}
                        # Includes a Future cancelled or broken before claiming
                        # its token: invalidate any residual queued work first.
                        with task.lock:
                            if task.state == "armed":
                                task.state = "aborted"
                                task.frame = None
                            self._release_task_resources(task)
                    self._entries[evidence_id].update(result)
                    completed.append(dict(result))
                    del self._futures[evidence_id]
                    del self._tasks[evidence_id]
        return completed

    @property
    def pending(self) -> bool:
        with self._lock:
            return bool(self._futures)

    def flush(self) -> list[dict]:
        with self._lock:
            futures = list(self._futures.values())
        wait(futures)
        self.poll()
        with self._lock:
            return [dict(value) for value in self._entries.values() if value["state"] != "saved"]

    def close(self) -> list[dict]:
        with self._lock:
            self._closed = True
        # Workers never need the store lock. Do not hold it while joining them.
        self._pool.shutdown(wait=True, cancel_futures=False)
        return self.poll()
