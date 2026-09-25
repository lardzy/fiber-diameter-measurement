"""Runtime receipts for lossless raster files already written or decoded.

Receipts share immutable pixel bytes with the live document. They are never
serialized and must not be made from a path merely because that path exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import shutil
import stat

from fdm.atomic_io import atomic_replace_file, staged_path_for
from fdm.raster import RasterPlane
from fdm.services.raster_io import RasterMetadata


@dataclass(frozen=True, slots=True)
class AssetFileStamp:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def read(cls, path: str | Path) -> AssetFileStamp | None:
        try:
            info = Path(path).stat()
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            return None
        return cls(info.st_dev, info.st_ino, info.st_size,
                   info.st_mtime_ns, info.st_ctime_ns)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class RasterAssetReceipt:
    path: Path
    plane: RasterPlane = field(repr=False)
    metadata: RasterMetadata | None
    stamp: AssetFileStamp
    sha256: str

    @classmethod
    def from_verified_file(
        cls,
        path: str | Path,
        plane: RasterPlane,
        metadata: RasterMetadata | None = None,
        *,
        expected_stamp: AssetFileStamp | None = None,
    ) -> RasterAssetReceipt | None:
        """Record a successful lossless write/read, not an unverified candidate.

        Readers pass the stamp captured BEFORE decoding so a concurrent file
        replacement cannot associate new file bytes with old in-memory pixels.
        Failure only disables reuse; the authoritative plane remains usable.
        """
        try:
            resolved = Path(path).resolve()
            before = AssetFileStamp.read(resolved)
            if before is None or (expected_stamp is not None and before != expected_stamp):
                return None
            digest = file_sha256(resolved)
            if AssetFileStamp.read(resolved) != before:
                return None
            return cls(resolved, plane, metadata, before, digest)
        except OSError:
            return None

    def matches(self, plane: RasterPlane, metadata: RasterMetadata | None) -> bool:
        # Identity is deliberate: RasterPlane is immutable, while comparing its
        # dataclass value or computing its hash would scan the entire image.
        return self.plane is plane and self.metadata == metadata

    def refreshed(self) -> RasterAssetReceipt | None:
        current = AssetFileStamp.read(self.path)
        if current is None:
            return None
        if current == self.stamp:
            return self
        try:
            if file_sha256(self.path) != self.sha256:
                return None
            if AssetFileStamp.read(self.path) != current:
                return None
        except OSError:
            return None
        return replace(self, stamp=current)


def copy_verified_raster_asset(
    source: RasterAssetReceipt, target: Path,
) -> RasterAssetReceipt:
    """Copy without encoding, validating the staged bytes before publication."""
    with staged_path_for(target, suffix=target.suffix) as staged:
        shutil.copyfile(source.path, staged)
        if file_sha256(staged) != source.sha256:
            raise OSError("复制图像资源的校验失败，源文件可能在复制期间发生变化。")
        atomic_replace_file(staged, target)
    stamp = AssetFileStamp.read(target)
    if stamp is None:
        raise OSError(f"无法确认已复制的图像资源: {target}")
    return replace(source, path=target.resolve(), stamp=stamp)
