from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO


def _fsync_directory_best_effort(directory: Path) -> None:
    """Best-effort durability barrier for a directory entry update."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError:
        # Directory fsync is unavailable on some supported platforms/filesystems.
        pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


@contextmanager
def staged_path_for(
    target: str | Path,
    *,
    suffix: str = ".tmp",
) -> Iterator[Path]:
    """Reserve a temporary path in the target directory and clean it on exit."""

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_replace_file(staged_path: str | Path, target: str | Path) -> Path:
    """Fsync and atomically replace *target* with a same-directory staged file."""

    source = Path(staged_path)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source.parent.resolve() != target_path.parent.resolve():
        raise ValueError("atomic replacement requires a staged file in the target directory")
    # Windows implements os.fsync() through the CRT _commit() call, which
    # rejects a descriptor opened read-only with EBADF.  The staged file is
    # ours and must be writable for replacement, so reopen it read/write for
    # the final durability barrier.
    with source.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(source, target_path)
    _fsync_directory_best_effort(target_path.parent)
    return target_path


def _atomic_write(
    target: str | Path,
    writer: Callable[[BinaryIO], None],
) -> Path:
    target_path = Path(target)
    with staged_path_for(target_path) as temporary_path:
        with temporary_path.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        return atomic_replace_file(temporary_path, target_path)


def atomic_write_bytes(target: str | Path, payload: bytes) -> Path:
    data = bytes(payload)
    return _atomic_write(target, lambda stream: stream.write(data))


def atomic_write_text(
    target: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    return atomic_write_bytes(target, str(text).encode(encoding))


def atomic_write_json(
    target: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
) -> Path:
    serialized = json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        indent=indent,
        allow_nan=False,
    )
    return atomic_write_text(target, serialized, encoding="utf-8")


def atomic_copy_file(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    try:
        if source_path.resolve() == target_path.resolve():
            return target_path
    except OSError:
        pass

    def copy_to(stream: BinaryIO) -> None:
        with source_path.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, stream)

    return _atomic_write(target_path, copy_to)
