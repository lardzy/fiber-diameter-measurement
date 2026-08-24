from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import ctypes
import json
import ntpath
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Lock
import time
from typing import BinaryIO

from fdm.atomic_io import atomic_replace_file, atomic_write_json, staged_path_for

if sys.platform.startswith("win"):
    import msvcrt
else:
    import fcntl


_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_MIN_FREE_RESERVE_BYTES = 512 * 1024 * 1024
_WINDOWS_DRIVE_REMOTE = 4
_READ_CACHE_PREFIX = "fdm-slide-cache-"
_OWNER_LOCK_NAME = ".owner.lock"
_OUTPUT_OWNER_LOCK_SUFFIX = ".owner.lock"
_OUTPUT_PUBLISHED_MARKER_SUFFIX = ".published.json"
_LEGACY_READ_CACHE_MIN_AGE_SECONDS = 24 * 60 * 60
_STARTUP_CLEANUP_GUARD = Lock()
_STARTUP_CLEANED_ROOTS: set[tuple[str, str]] = set()


class DigitalSlideCacheCancelled(RuntimeError):
    """Raised when the user cancels a network-slide localization."""


def _try_acquire_owner_lock(path: Path) -> BinaryIO | None:
    """Acquire a one-byte cross-process lock, returning its live stream."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stream: BinaryIO | None = None
    try:
        stream = path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if sys.platform.startswith("win"):
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except OSError:
        if stream is not None:
            stream.close()
        return None


def _release_owner_lock(stream: BinaryIO | None) -> None:
    if stream is None:
        return
    try:
        stream.seek(0)
        if sys.platform.startswith("win"):
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        stream.close()


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _windows_drive_type(root: str) -> int | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
        get_drive_type.argtypes = [ctypes.c_wchar_p]
        get_drive_type.restype = ctypes.c_uint
        return int(get_drive_type(root))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def is_network_file_path(path: str | Path) -> bool:
    """Return whether *path* is a UNC path or a Windows mapped network drive."""

    token = os.fspath(path).strip()
    if token.startswith(("\\\\", "//")):
        return True
    if not sys.platform.startswith("win"):
        return False
    drive, _tail = ntpath.splitdrive(token)
    if not drive:
        return False
    return _windows_drive_type(f"{drive}\\") == _WINDOWS_DRIVE_REMOTE


class DigitalSlideSessionCache:
    """Local mirrors and staging files for network digital slides.

    A ``.fdmslide`` file is a SQLite tile database.  Random SQLite reads over
    SMB are both latency-sensitive and dependent on remote filesystem locking.
    This cache performs one sequential, read-only copy and lets every viewport
    render use the local mirror.  Network captures are also built locally and
    published with one sequential copy only after SQLite has been closed.
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        output_staging_root: str | Path | None = None,
        temporary_parent: str | Path | None = None,
        network_path_predicate: Callable[[str | Path], bool] = is_network_file_path,
    ) -> None:
        self._network_path_predicate = network_path_predicate
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._root = Path(root).expanduser() if root is not None else None
        self._temporary_parent = (
            Path(temporary_parent).expanduser()
            if temporary_parent is not None
            else Path(tempfile.gettempdir())
        )
        self._read_cache_lock: BinaryIO | None = None
        self._read_cache_lock_path: Path | None = None
        self._output_staging_root = (
            Path(output_staging_root).expanduser()
            if output_staging_root is not None
            else None
        )
        self._owned_output_paths: set[Path] = set()
        self._output_locks: dict[Path, BinaryIO] = {}

    def cleanup_abandoned_once(self) -> tuple[int, int]:
        """Clean abandoned disposable files once per process and root pair."""

        staging_root = self._output_staging_root
        key = (
            os.path.normcase(str(self._temporary_parent.resolve(strict=False))),
            os.path.normcase(str(staging_root.resolve(strict=False)))
            if staging_root is not None
            else "",
        )
        with _STARTUP_CLEANUP_GUARD:
            if key in _STARTUP_CLEANED_ROOTS:
                return 0, 0
            _STARTUP_CLEANED_ROOTS.add(key)
        return self.cleanup_abandoned()

    def cleanup_abandoned(self) -> tuple[int, int]:
        """Remove stale read caches and safely published capture staging.

        Unpublished capture files are deliberately preserved because they may
        be the only recoverable result of a forced shutdown or network outage.
        """

        return (
            self.cleanup_abandoned_read_caches(),
            self.cleanup_abandoned_published_outputs(),
        )

    def cleanup_abandoned_read_caches(self) -> int:
        parent = self._temporary_parent
        if not parent.is_dir():
            return 0
        removed = 0
        now = time.time()
        current_root = (
            self._root.resolve(strict=False)
            if self._root is not None
            else None
        )
        try:
            candidates = tuple(parent.glob(f"{_READ_CACHE_PREFIX}*"))
        except OSError:
            return 0
        for candidate in candidates:
            try:
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                if current_root is not None and candidate.resolve(strict=False) == current_root:
                    continue
                lock_path = candidate / _OWNER_LOCK_NAME
                if not lock_path.exists():
                    age_seconds = max(0.0, now - candidate.stat().st_mtime)
                    if age_seconds < _LEGACY_READ_CACHE_MIN_AGE_SECONDS:
                        continue
                lock = _try_acquire_owner_lock(lock_path)
                if lock is None:
                    continue
                _release_owner_lock(lock)
                shutil.rmtree(candidate)
                removed += 1
            except OSError:
                continue
        return removed

    def cleanup_abandoned_published_outputs(self) -> int:
        root = self._output_staging_root
        if root is None or not root.is_dir():
            return 0
        removed = 0
        try:
            markers = tuple(
                root.glob(
                    f"capture-*.fdmslide{_OUTPUT_PUBLISHED_MARKER_SUFFIX}"
                )
            )
        except OSError:
            return 0
        for marker_path in markers:
            try:
                payload = json.loads(marker_path.read_text(encoding="utf-8"))
                marker_name = marker_path.name
                working_name = marker_name[: -len(_OUTPUT_PUBLISHED_MARKER_SUFFIX)]
                working_path = marker_path.with_name(working_name)
                if (
                    not isinstance(payload, dict)
                    or int(payload.get("version", 0)) != 1
                    or str(payload.get("working_name", "")) != working_name
                    or not working_name.startswith("capture-")
                    or not working_name.endswith(".fdmslide")
                    or working_path.is_symlink()
                ):
                    continue
                expected_size = int(payload.get("size", -1))
                if working_path.exists() and working_path.stat().st_size != expected_size:
                    # A changed local copy may contain recovery-only data that
                    # was never included in the previously published target.
                    continue
                lock_path = self._output_lock_path(working_path)
                lock = _try_acquire_owner_lock(lock_path)
                if lock is None:
                    continue
                _release_owner_lock(lock)
                existed = working_path.exists()
                self._delete_sqlite_files(working_path)
                marker_path.unlink(missing_ok=True)
                lock_path.unlink(missing_ok=True)
                if existed:
                    removed += 1
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return removed

    def requires_local_copy(self, source: str | Path) -> bool:
        return bool(self._network_path_predicate(source))

    def localize(
        self,
        source: str | Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> Path:
        source_path = Path(source).expanduser()
        if not self.requires_local_copy(source_path):
            return source_path.resolve(strict=False)
        if cancellation_requested is not None and cancellation_requested():
            raise DigitalSlideCacheCancelled("已取消复制网络数字化切片。")
        try:
            before = source_path.stat()
        except OSError as exc:
            raise OSError(f"无法访问网络数字化切片：{source_path}\n{exc}") from exc
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        self._reject_live_sqlite_sidecars(source_path)

        root = self._cache_root()
        total = max(0, int(before.st_size))
        try:
            available = shutil.disk_usage(root).free
        except OSError as exc:
            raise OSError(f"无法检查本机临时目录剩余空间：{root}\n{exc}") from exc
        required = total + _MIN_FREE_RESERVE_BYTES
        if available < required:
            raise OSError(
                "本机临时目录空间不足，无法缓存网络数字化切片："
                f"需要约 {required / (1024**3):.2f} GiB，"
                f"当前可用 {available / (1024**3):.2f} GiB。"
            )

        fingerprint = self._fingerprint(source_path, before.st_size, before.st_mtime_ns)
        target = root / f"{fingerprint}.fdmslide"
        if target.is_file() and target.stat().st_size == total:
            if progress_callback is not None:
                progress_callback(total, total)
            return target

        copied = 0
        with staged_path_for(target, suffix=".fdmslide.part") as staged_path:
            try:
                with source_path.open("rb") as source_file, staged_path.open("wb") as target_file:
                    while True:
                        if cancellation_requested is not None and cancellation_requested():
                            raise DigitalSlideCacheCancelled("已取消复制网络数字化切片。")
                        payload = source_file.read(_COPY_CHUNK_BYTES)
                        if not payload:
                            break
                        target_file.write(payload)
                        copied += len(payload)
                        if progress_callback is not None:
                            progress_callback(copied, total)
                    target_file.flush()
                    os.fsync(target_file.fileno())
            except DigitalSlideCacheCancelled:
                raise
            except OSError as exc:
                raise OSError(f"复制网络数字化切片失败：{source_path}\n{exc}") from exc

            after = source_path.stat()
            self._reject_live_sqlite_sidecars(source_path)
            if (
                int(after.st_size) != int(before.st_size)
                or int(after.st_mtime_ns) != int(before.st_mtime_ns)
            ):
                raise OSError("网络数字化切片在复制期间发生变化，请等待写入结束后重试。")
            if staged_path.stat().st_size != total:
                raise OSError("网络数字化切片复制不完整，请检查网络连接后重试。")
            atomic_replace_file(staged_path, target)

        if progress_callback is not None:
            progress_callback(total, total)
        return target

    def working_output_path(
        self,
        target: str | Path,
        *,
        expected_bytes: int = 0,
        reserve_bytes: int = _MIN_FREE_RESERVE_BYTES,
    ) -> Path:
        """Return a local capture path when *target* is on a network share.

        The returned network-capture path is unique and intentionally lives in
        a persistent local staging directory when one was configured.  The
        caller may therefore retain it as a recovery copy if publishing fails.
        """

        target_path = Path(target).expanduser()
        if not self.requires_local_copy(target_path):
            return target_path.resolve(strict=False)

        root = self._output_root()
        try:
            available = shutil.disk_usage(root).free
        except OSError as exc:
            raise OSError(f"无法检查本机切片暂存目录剩余空间：{root}\n{exc}") from exc
        required = max(0, int(expected_bytes)) + max(0, int(reserve_bytes))
        if available < required:
            raise OSError(
                "本机切片暂存目录空间不足："
                f"需要约 {required / (1024**3):.2f} GiB，"
                f"当前可用 {available / (1024**3):.2f} GiB。"
            )

        safe_stem = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in target_path.stem
        ).strip("_")[:48] or "digital-slide"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=root,
            prefix=f"capture-{safe_stem}-",
            suffix=".fdmslide",
        )
        os.close(descriptor)
        working_path = Path(temporary_name)
        working_path.unlink(missing_ok=True)
        lock_path = self._output_lock_path(working_path)
        owner_lock = _try_acquire_owner_lock(lock_path)
        if owner_lock is None:
            lock_path.unlink(missing_ok=True)
            raise OSError(f"无法锁定本机切片暂存文件：{working_path}")
        self._owned_output_paths.add(working_path)
        self._output_locks[working_path] = owner_lock
        return working_path

    def publish(
        self,
        local_source: str | Path,
        target: str | Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Publish a closed local SQLite slide to *target* atomically.

        The existing target is preserved until the complete staged copy has
        been flushed.  The local source is never deleted by this operation.
        """

        source_path = Path(local_source).expanduser()
        target_path = Path(target).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        self._reject_live_sqlite_sidecars(source_path)
        before = source_path.stat()
        total = max(0, int(before.st_size))
        copied = 0
        try:
            with staged_path_for(target_path, suffix=".fdmslide.publish") as staged_path:
                with source_path.open("rb") as source_file, staged_path.open("wb") as target_file:
                    while True:
                        payload = source_file.read(_COPY_CHUNK_BYTES)
                        if not payload:
                            break
                        target_file.write(payload)
                        copied += len(payload)
                        if progress_callback is not None:
                            progress_callback(copied, total)
                    target_file.flush()
                    os.fsync(target_file.fileno())

                after = source_path.stat()
                self._reject_live_sqlite_sidecars(source_path)
                if (
                    int(after.st_size) != int(before.st_size)
                    or int(after.st_mtime_ns) != int(before.st_mtime_ns)
                ):
                    raise OSError("本机数字化切片在发布期间发生变化，已保留原网络文件。")
                if staged_path.stat().st_size != total:
                    raise OSError("数字化切片发布不完整，已保留原网络文件。")
                atomic_replace_file(staged_path, target_path)
        except OSError as exc:
            raise OSError(f"无法将数字化切片发布到网络目录：{target_path}\n{exc}") from exc

        self.mark_output_published(source_path, target_path)
        if progress_callback is not None:
            progress_callback(total, total)
        return target_path

    def mark_output_published(
        self,
        local_source: str | Path,
        published_target: str | Path,
    ) -> bool:
        """Persist that a staging file is now a disposable published mirror."""

        source_path = Path(local_source).expanduser()
        if source_path not in self._owned_output_paths or not source_path.is_file():
            return False
        marker_path = self._published_marker_path(source_path)
        try:
            atomic_write_json(
                marker_path,
                {
                    "version": 1,
                    "working_name": source_path.name,
                    "published_target": str(Path(published_target).expanduser()),
                    "size": int(source_path.stat().st_size),
                    "published_at_ns": time.time_ns(),
                },
            )
        except OSError:
            # Publication itself already succeeded. A missing cleanup marker
            # may leave a harmless duplicate after a crash, but must not turn
            # a successful network publish into a reported data failure.
            return False
        return True

    def retain_output(self, path: str | Path) -> Path:
        """Keep a staged capture as a user-recoverable local file."""

        output_path = Path(path).expanduser()
        if output_path not in self._owned_output_paths:
            return output_path
        self._owned_output_paths.discard(output_path)
        _unlink_best_effort(self._published_marker_path(output_path))
        self._release_output_lock(output_path)
        return output_path

    def forget_output(self, path: str | Path) -> None:
        output_path = Path(path).expanduser()
        if output_path not in self._owned_output_paths:
            return
        self._owned_output_paths.discard(output_path)
        _unlink_best_effort(self._published_marker_path(output_path))
        self._release_output_lock(output_path)

    def cleanup(self) -> None:
        for output_path in tuple(self._owned_output_paths):
            self._delete_sqlite_files(output_path)
            _unlink_best_effort(self._published_marker_path(output_path))
            self._release_output_lock(output_path)
        self._owned_output_paths.clear()
        temporary = self._temporary
        self._temporary = None
        read_cache_lock = self._read_cache_lock
        read_cache_lock_path = self._read_cache_lock_path
        self._read_cache_lock = None
        self._read_cache_lock_path = None
        _release_owner_lock(read_cache_lock)
        if temporary is not None:
            temporary.cleanup()
            self._root = None
        elif read_cache_lock_path is not None:
            _unlink_best_effort(read_cache_lock_path)

    def _cache_root(self) -> Path:
        if self._root is None:
            self._temporary_parent.mkdir(parents=True, exist_ok=True)
            self._temporary = tempfile.TemporaryDirectory(
                prefix=_READ_CACHE_PREFIX,
                dir=self._temporary_parent,
            )
            self._root = Path(self._temporary.name)
        self._root.mkdir(parents=True, exist_ok=True)
        if self._read_cache_lock is None:
            lock_path = self._root / _OWNER_LOCK_NAME
            owner_lock = _try_acquire_owner_lock(lock_path)
            if owner_lock is None:
                raise OSError(f"无法锁定本机数字化切片缓存目录：{self._root}")
            self._read_cache_lock = owner_lock
            self._read_cache_lock_path = lock_path
        return self._root

    def _output_root(self) -> Path:
        root = self._output_staging_root
        if root is None:
            root = self._cache_root()
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _fingerprint(source: Path, size: int, mtime_ns: int) -> str:
        identity = f"{os.path.normcase(os.fspath(source))}\0{int(size)}\0{int(mtime_ns)}"
        return sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()[:24]

    @staticmethod
    def _reject_live_sqlite_sidecars(source: Path) -> None:
        active = [
            Path(f"{source}{suffix}")
            for suffix in ("-wal", "-shm", "-journal")
        ]
        if any(candidate.exists() for candidate in active):
            raise OSError(
                "网络数字化切片仍有 SQLite 写入侧文件；"
                "请先结束生成该切片的采集或写入程序，再重新打开。"
            )

    @staticmethod
    def _output_lock_path(path: Path) -> Path:
        return Path(f"{path}{_OUTPUT_OWNER_LOCK_SUFFIX}")

    @staticmethod
    def _published_marker_path(path: Path) -> Path:
        return Path(f"{path}{_OUTPUT_PUBLISHED_MARKER_SUFFIX}")

    def _release_output_lock(self, path: Path) -> None:
        owner_lock = self._output_locks.pop(path, None)
        _release_owner_lock(owner_lock)
        _unlink_best_effort(self._output_lock_path(path))

    @staticmethod
    def _delete_sqlite_files(path: Path) -> None:
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
            Path(f"{path}-journal"),
        ):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "DigitalSlideCacheCancelled",
    "DigitalSlideSessionCache",
    "is_network_file_path",
]
