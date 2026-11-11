"""Ownership-checked, asynchronous collection of disposable trial images.

Only files whose inode was recorded by this process before encoding are owned.
Project assets and caller-supplied EvidenceStore directories are never collected.
Closing a view is deliberately not a disposal operation.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
import uuid
import weakref

log = logging.getLogger(__name__)
_APP = "fdm.trial-evidence.v1"
_UUID = re.compile(r"^[0-9a-f]{32}$")
_FILE = re.compile(r"^[0-9a-f]{32}/[0-9a-f]{64}\.(png|jpg|pending)$")


def _ordinary(path: Path, *, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ValueError(f"不是普通{'目录' if directory else '文件'}：{path.name}")
    return info


def _identity(info):
    return [int(info.st_dev), int(info.st_ino)]


def _session_paths(session):
    """Read immutable record/history references without serializing geometry."""
    overrides = dict(session._evidence_overrides)
    paths = set()
    def record(value):
        if value is None:
            return
        evidence = overrides.get(value.id, value.evidence) or {}
        if evidence.get("path"):
            paths.add(str(Path(evidence["path"]).absolute()))
    def change(value):
        if value.collection == "measurements":
            record(value.before)
            record(value.after)
        elif value.collection == "batch":
            for child in value.after:
                change(child)
    for value in tuple(session.measurements.values()):
        record(value)
    for value in (*tuple(session._undo), *tuple(session._redo)):
        change(value)
    return paths


class _FileLock:
    def __init__(self, path):
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(path, flags, 0o600)
        self.locked = False
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("临时图片实例锁无效")
            if os.fstat(self.fd).st_size == 0:
                os.write(self.fd, b"0")
            os.lseek(self.fd, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.locked = True
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@dataclass
class CleanupReport:
    deleted_files: int = 0
    deleted_bytes: int = 0
    in_use: int = 0
    skipped: int = 0
    deferred: bool = False
    stopped: bool = False
    errors: list[str] = field(default_factory=list)

    def message(self):
        text = f"已清理 {self.deleted_files} 个临时文件，释放 {self.deleted_bytes / 1048576:.2f} MiB；使用中 {self.in_use} 个。"
        if self.deferred:
            text += " 正在测量或保存，剩余清理将稍后继续。"
        if self.stopped:
            text += " 清理已停止，剩余临时图片将在下次启动时处理。"
        if self.skipped or self.errors:
            text += f" 跳过 {self.skipped} 项；无法确认归属或暂不可删除的文件保持原样。"
        return text


@dataclass(frozen=True)
class CleanupProgress:
    """Cheap observer snapshot; totals count completed passes exactly once."""
    completed: int
    deleted_files: int
    deleted_bytes: int
    skipped: int
    last_report: CleanupReport
    busy: bool


class ResourceLease:
    def __init__(self, owner, token, paths=None):
        self.owner, self.token = owner, token
        self.paths = paths or {}

    def resolve(self, path):
        """Path captured with this read lease; None means the trial was retired."""
        return self.paths.get(str(Path(path).absolute()), str(Path(path).absolute()))

    def close(self):
        owner, self.owner = self.owner, None
        if owner is not None:
            owner._release(self.token)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


class EvidenceResources:
    def __init__(self, root=None):
        # Resolving only the OS temporary parent allows macOS /var -> /private/var.
        self.root = (Path(root).absolute() if root is not None else
                     Path(tempfile.gettempdir()).resolve() / "fdm-trial-evidence-v1")
        self.instance = uuid.uuid4().hex
        self.directory = self.root / self.instance
        self._lock = threading.RLock()
        self._disk_lock = threading.RLock()
        self._instance_lock = None
        self._stores = {}
        self._files = {}  # absolute path -> recorded inode, store ID
        self._eligible = set()
        self._leases = {}
        self._published = {}
        self._retiring = {}
        self._deleted = set()
        self._collection_offsets = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdm-evidence-cleanup")
        self._future = None
        self._rerun = False
        self._deferred = False
        self._shutdown = False
        self._stop_cleanup = threading.Event()
        self._accounted_futures = weakref.WeakSet()
        self._future_numbers = weakref.WeakKeyDictionary()
        self._scheduled_passes = 0
        self._last_report_number = 0
        self._cleanup_handoff = False
        self._completed_passes = 0
        self._deleted_files_total = 0
        self._deleted_bytes_total = 0
        self._skipped_total = 0
        self._last_report = CleanupReport()

    def register_store(self, store):
        key = uuid.uuid4().hex
        with self._lock:
            self._stores[key] = {"store": weakref.ref(store), "project": None,
                                 "session": None, "discarded": False}
        return key, self.directory / key

    def attach_owner(self, key, project, session):
        with self._lock:
            self._stores[key].update(project=project, session=session)

    def activity(self):
        return self._lease(None)

    def pin_paths(self, paths, *, resolve=True):
        token = uuid.uuid4().hex
        resolved = {}
        with self._lock:
            for raw in paths:
                if not raw:
                    continue
                original = target = str(Path(raw).absolute())
                visited = set()
                while resolve and target in self._published and target not in visited:
                    visited.add(target)
                    target = self._published[target]
                claim = self._retiring.get(target)
                if claim is not None:
                    if claim["phase"] == "checking":
                        claim["cancelled"] = True
                    else:
                        target = None
                if target in self._deleted:
                    target = None
                resolved[original] = target
            self._leases[token] = frozenset(path for path in resolved.values() if path)
        lease = ResourceLease(self, token, resolved)
        if not resolve and any(path is None for path in resolved.values()):
            lease.close()
            raise ValueError("保存快照引用了已经放弃的临时图片")
        return lease

    def pin_payload(self, payload):
        paths = []
        for session in payload.get("live_experiments", ()):
            records = session.get("measurements", ())
            for record in records.values() if isinstance(records, dict) else records:
                value = record.get("evidence") or {}
                if value.get("path"):
                    paths.append(value["path"])
        return self.pin_paths(paths, resolve=False)

    def _lease(self, paths):
        token = uuid.uuid4().hex
        with self._lock:
            self._leases[token] = paths
        return ResourceLease(self, token)

    def _release(self, token):
        with self._lock:
            self._leases.pop(token, None)
            # A still-undoable image may remain eligible for hours. Do not
            # rescan the whole journal after every subsequent camera image.
            retry = not self._shutdown and self._deferred
        if retry:
            self.request_cleanup()

    def _ensure_directory(self, key):
        # Encoding worker only. No directory IO is done by reserve or the UI.
        with self._disk_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            _ordinary(self.root, directory=True)
            if self._instance_lock is None:
                self.directory.mkdir()
                self._instance_lock = _FileLock(self.directory / "instance.lock")
                owner = {"app": _APP, "instance": self.instance, "pid": os.getpid(),
                         "directory_identity": _identity(_ordinary(self.directory, directory=True))}
                (self.directory / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
            folder = self.directory / key
            folder.mkdir(exist_ok=True)
            _ordinary(folder, directory=True)

    def open_pending(self, key, destination):
        self._ensure_directory(key)
        destination = Path(destination)
        relative = destination.relative_to(self.directory).as_posix()
        if not _FILE.fullmatch(relative) or relative.split("/")[0] != key:
            raise ValueError("临时图片路径不属于此存储实例")
        pending = destination.with_suffix(".pending")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("临时图片目标已存在，未覆盖")
        handle = pending.open("xb", buffering=0)
        try:
            identity = _identity(os.fstat(handle.fileno()))
            claims = [dict(path=path.relative_to(self.directory).as_posix(), identity=identity)
                      for path in (pending, destination)]
            with self._disk_lock:
                journal = self.directory / "files.jsonl"
                with journal.open("a", encoding="utf-8") as stream:
                    for claim in claims:
                        stream.write(json.dumps(claim) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            with self._lock:
                for path in (pending, destination):
                    self._files[str(path)] = (identity, key)
                    if self._stores[key]["discarded"]:
                        self._eligible.add(str(path))
            return handle
        except BaseException:
            handle.close()
            # No successful claim means this file is not collected automatically.
            raise

    def published_paths(self, paths):
        """Called only after project publication and live/history rebasing."""
        paths = {str(Path(old).absolute()): str(Path(new).absolute())
                 for old, new in paths.items() if old and old != new}
        with self._lock:
            self._published.update(paths)
            stores = tuple(item["store"]() for item in self._stores.values())
        for store in stores:
            if store is not None:
                store.rebase_owned_paths(paths)
        collect = False
        with self._lock:
            for old_path in paths:
                if old_path in self._files:
                    self._eligible.add(old_path)
                    self._eligible.add(str(Path(old_path).with_suffix(".pending")))
                    collect = True
        if collect:
            self.request_cleanup()

    def discard_session(self, session):
        self._discard(lambda item: item["session"] is session)

    def discard_project(self, project):
        self._discard(lambda item: item["project"] is project)

    def _discard(self, matches):
        with self._lock:
            keys = set()
            for key, item in self._stores.items():
                if matches(item):
                    item.update(discarded=True, session=None, project=None)
                    keys.add(key)
            self._eligible.update(path for path, (_, key) in self._files.items() if key in keys)
        self.request_cleanup()

    def request_cleanup(self):
        with self._lock:
            if self._shutdown:
                completed = Future()
                completed.set_result(CleanupReport(stopped=True))
                return completed
            if self._future is not None and not self._future.done():
                self._rerun = True
                return self._future
            self._future = self._pool.submit(self._collect)
            self._scheduled_passes += 1
            self._future_numbers[self._future] = self._scheduled_passes
            self._future.add_done_callback(self._cleanup_finished)
            return self._future

    def cleanup_progress(self):
        """Observe deferred automatic continuations without scheduling new IO."""
        with self._lock:
            last = self._last_report
            report = CleanupReport(last.deleted_files, last.deleted_bytes, last.in_use,
                                   last.skipped, last.deferred, last.stopped, list(last.errors))
            if self._shutdown:
                report.stopped = True
            busy = (not self._shutdown and (self._cleanup_handoff or self._deferred
                    or self._rerun or self._future is not None
                    and self._future not in self._accounted_futures))
            return CleanupProgress(self._completed_passes, self._deleted_files_total,
                                   self._deleted_bytes_total, self._skipped_total, report, bool(busy))

    def _cleanup_finished(self, completed):
        # A reader/writer may release its lease after _collect decides to stop,
        # but before Future.done becomes true. Preserve that wakeup.
        try:
            report = completed.result()
        except BaseException as exc:
            report = CleanupReport(stopped=completed.cancelled(), errors=[str(exc)])
        with self._lock:
            self._accounted_futures.add(completed)
            self._completed_passes += 1
            self._deleted_files_total += report.deleted_files
            self._deleted_bytes_total += report.deleted_bytes
            self._skipped_total += report.skipped
            number = self._future_numbers.get(completed, 0)
            if number >= self._last_report_number:
                self._last_report = report
                self._last_report_number = number
            repeat = self._future is completed and self._rerun and not self._shutdown
            if repeat:
                self._future = None
                self._rerun = False
                self._cleanup_handoff = True
        if repeat:
            try:
                self.request_cleanup()
            finally:
                with self._lock:
                    self._cleanup_handoff = False

    def _protected(self, path):
        with self._lock:
            if any(value is None for value in self._leases.values()):
                self._deferred = True
                return "busy"
            if any(path in value for value in self._leases.values() if value is not None):
                self._deferred = True
                return "pinned"
        return None

    def _delete(self, directory, relative, identity, report, *, active=False):
        if self._stop_cleanup.is_set():
            report.stopped = True
            return
        if not isinstance(relative, str) or not _FILE.fullmatch(relative):
            report.skipped += 1
            return
        path = directory / relative
        key = str(path)
        with self._lock:
            # A busy pause deliberately retries its boundary journal row. A
            # previously deleted row is not an "in use" image on that retry.
            if key in self._deleted:
                return
            reason = self._protected(key)
            if reason:
                report.in_use += 1
                report.deferred |= reason == "busy"
                return
            if active and key not in self._eligible:
                report.in_use += int(path.suffix != ".pending")
                return
            claim = {"phase": "checking", "cancelled": False}
            self._retiring[key] = claim
        try:
            # No filesystem call is made while holding the application mutex.
            # A reader arriving during these checks cancels this retirement.
            _ordinary(directory, directory=True)
            _ordinary(path.parent, directory=True)
            info = _ordinary(path)
            if _identity(info) != identity or info.st_nlink != 1:
                report.skipped += 1
                return
            with self._lock:
                protected = self._protected(key)
                if self._stop_cleanup.is_set():
                    report.stopped = True
                    return
                if claim["cancelled"] or protected:
                    report.in_use += 1
                    report.deferred |= protected == "busy"
                    self._deferred = True
                    return
                claim["phase"] = "deleting"
            # Later readers resolve a published asset, or reject a stale read
            # from an already-discarded trial; they never wait on this unlink.
            path.unlink()
            report.deleted_files += 1
            report.deleted_bytes += info.st_size
            with self._lock:
                self._eligible.discard(key)
                self._deleted.add(key)
        except FileNotFoundError:
            with self._lock:
                self._eligible.discard(key)
                self._deleted.add(key)
        except (OSError, ValueError) as exc:
            report.skipped += 1
            if len(report.errors) < 20:
                report.errors.append(str(exc))
        finally:
            with self._lock:
                self._retiring.pop(key, None)

    def _collect_instance(self, directory, report, *, active=False):
        lock = None
        try:
            if self._stop_cleanup.is_set():
                report.stopped = True
                return
            info = _ordinary(directory, directory=True)
            owner_file = directory / "owner.json"
            _ordinary(owner_file)
            if owner_file.stat().st_size > 4096:
                raise ValueError("临时图片归属清单过大")
            owner = json.loads(owner_file.read_text(encoding="utf-8"))
            if (owner.get("app") != _APP or owner.get("instance") != directory.name
                    or owner.get("directory_identity") != _identity(info)):
                raise ValueError("临时图片目录身份不匹配")
            lock_file = directory / "instance.lock"
            _ordinary(lock_file)
            if not active:
                try:
                    lock = _FileLock(lock_file)
                except OSError:
                    report.in_use += 1
                    return
            journal = directory / "files.jsonl"
            journal_info = _ordinary(journal)
            live_paths = set()
            if active:
                with self._lock:
                    sessions = {id(item["session"]): item["session"] for item in self._stores.values()
                                if item["session"] is not None}
                for session in sessions.values():
                    live_paths.update(_session_paths(session))
            # Resume a busy-paused pass at its byte offset. Parsing the entire
            # growing journal after every measurement completion caused avoidable
            # CPU/IO competition even though deletion itself ran off-thread.
            stamp = tuple(_identity(journal_info))
            previous = self._collection_offsets.get(str(directory))
            if previous is not None and previous[0] == stamp and previous[1] <= journal_info.st_size:
                offset, folders = previous[1], previous[2]
            else:
                offset, folders = 0, set()
            with journal.open("rb") as stream:
                stream.seek(offset)
                processed = 0
                while True:
                    if self._stop_cleanup.is_set():
                        report.stopped = True
                        return
                    if self._protected("") == "busy":
                        self._collection_offsets[str(directory)] = (stamp, stream.tell(), folders)
                        report.deferred = True
                        return
                    position = stream.tell()
                    line = stream.readline(4097)
                    if not line:
                        break
                    if len(line) > 4096:
                        raise ValueError("临时图片文件清单无效")
                    if not line.endswith(b"\n"):
                        # An active writer can have appended only part of its
                        # claim. Retry from that line; never parse its fragment.
                        self._collection_offsets[str(directory)] = (stamp, position, folders)
                        return
                    try:
                        entry = json.loads(line)
                        relative, identity = entry.get("path", ""), entry.get("identity")
                        if not _FILE.fullmatch(relative):
                            report.skipped += 1
                            continue
                    except (ValueError, TypeError, AttributeError):
                        report.skipped += 1
                        continue
                    folders.add(relative.split("/")[0])
                    if str(directory / relative) in live_paths:
                        report.in_use += 1
                    else:
                        self._delete(directory, relative, identity, report, active=active)
                    if report.deferred or self._protected("") == "busy":
                        self._collection_offsets[str(directory)] = (stamp, position, folders)
                        report.deferred = True
                        return
                    processed += 1
                    if processed % 2 == 0:
                        time.sleep(.020)  # Small metadata bursts; leave time for native painting.
            self._collection_offsets.pop(str(directory), None)
            # rmdir never removes unknown files; no recursive delete is used.
            for key in folders:
                if self._stop_cleanup.is_set():
                    report.stopped = True
                    return
                folder = directory / key
                try:
                    _ordinary(folder, directory=True)
                    folder.rmdir()
                except (OSError, ValueError):
                    pass
            if not active:
                remaining = {item.name for item in directory.iterdir()}
                if remaining <= {"owner.json", "files.jsonl", "instance.lock"}:
                    # Remove the owner receipt first: another collector cannot
                    # accept this directory after the Windows lock is released.
                    for name in ("owner.json", "files.jsonl"):
                        target = directory / name
                        _ordinary(target)
                        target.unlink()
                    lock.close()
                    lock = None
                    _ordinary(lock_file)
                    lock_file.unlink()
                    directory.rmdir()
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            report.skipped += 1
            if len(report.errors) < 20:
                report.errors.append(str(exc))
        finally:
            if lock is not None:
                lock.close()

    def _collect(self):
        report = CleanupReport()
        while True:
            if self._stop_cleanup.is_set():
                report.stopped = True
                break
            with self._lock:
                self._rerun = False
                self._deferred = False
            try:
                _ordinary(self.root, directory=True)
                for path in self.root.iterdir():
                    if self._stop_cleanup.is_set():
                        report.stopped = True
                        break
                    if not _UUID.fullmatch(path.name):
                        continue
                    if self._protected("") == "busy":
                        report.deferred = True
                        break
                    self._collect_instance(path, report, active=path == self.directory)
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as exc:
                report.errors.append(str(exc))
            with self._lock:
                if not self._rerun or self._deferred:
                    break
        if report.errors:
            log.info("Trial image cleanup skipped files: %s", report.errors)
        return report

    def stop_cleanup(self):
        """Application aboutToQuit hook: stop maintenance without joining IO.

        Keep the instance lock until OS process teardown: an asynchronous image
        reader can still own a lease after the last Qt window has closed.
        """
        self._stop_cleanup.set()
        with self._lock:
            self._shutdown = True
            self._rerun = False
        self._pool.shutdown(wait=False, cancel_futures=True)

    def shutdown(self):
        """Tests/process teardown only; regular view close must not call this."""
        self.stop_cleanup()
        self._pool.shutdown(wait=True)
        if self._instance_lock is not None:
            self._instance_lock.close()
            self._instance_lock = None


_resources = None


def get_evidence_resources():
    global _resources
    if _resources is None:
        _resources = EvidenceResources()
    return _resources
