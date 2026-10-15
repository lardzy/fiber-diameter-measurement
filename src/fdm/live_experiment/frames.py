"""Owned image frames and byte-bounded leases for live measurements.

No Qt objects cross the compute boundary. A source transfers ownership or makes
one snapshot; consumers retain the same immutable array, never a mutable slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any
import uuid

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameHandle:
    source_id: str
    generation: int
    sequence: int
    received_at: float
    pixels: np.ndarray = field(repr=False, compare=False)
    pixel_format: str = "BGR"
    live: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def snapshot(cls, pixels, *, source_id: str, generation: int = 0,
                 sequence: int = 0, received_at: float | None = None,
                 pixel_format: str | None = None, live: bool = True,
                 transfer_ownership: bool = False, metadata=None) -> "FrameHandle":
        array = np.asarray(pixels)
        if array.dtype != np.uint8 or array.ndim not in (2, 3):
            raise ValueError("实时测量帧必须是 uint8 灰度、BGR 或 BGRA 图像。")
        if array.ndim == 3 and array.shape[2] not in (3, 4):
            raise ValueError("实时测量帧通道数必须为 1、3 或 4。")
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("图像不能为空。")
        # Ownership transfer is exclusively for source adapters allocating a
        # fresh buffer, never shared-memory ring slots or caller-owned images.
        if not transfer_ownership or not array.flags.owndata or not array.flags.c_contiguous:
            array = np.array(array, copy=True, order="C")
        array.setflags(write=False)
        expected = "GRAY" if array.ndim == 2 else ("BGR" if array.shape[2] == 3 else "BGRA")
        if pixel_format is not None and pixel_format != expected:
            raise ValueError("像素格式与帧形状不一致。")
        stamp = time.perf_counter() if received_at is None else float(received_at)
        if not math.isfinite(stamp):
            raise ValueError("帧时间必须有限。")
        return cls(str(source_id), int(generation), int(sequence), stamp,
                   array, expected, live, dict(metadata or {}))

    @property
    def frame_id(self) -> str:
        return f"{self.source_id}:{self.generation}:{self.sequence}"

    @property
    def size(self) -> tuple[int, int]:
        return self.pixels.shape[1], self.pixels.shape[0]

    @property
    def nbytes(self) -> int:
        return self.pixels.nbytes


@dataclass(frozen=True, slots=True)
class PresentedFrameToken:
    frame: FrameHandle
    field_epoch: int
    display_revision: int
    # affine scale and offset; captured after painting, not on frame arrival
    scale: float
    offset: tuple[float, float]

    def image_point(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.offset[0]) / self.scale,
                (y - self.offset[1]) / self.scale)


class FrameBudgetExceeded(RuntimeError):
    pass


def acquisition_reservation(frame: FrameHandle) -> int:
    """Source ring/handoff allowance shared by preview and evidence admission."""
    if not frame.live:
        return 0
    raw = frame.metadata.get("acquisition_reserved_bytes", frame.nbytes)
    return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else frame.nbytes


class FrameBudget:
    """Logical pixel/source/codec budget; not a bound on total process RSS.

    Pixel owners count once. Source rings share their largest reservation.
    Scratch groups are independent serial executors: each group's largest live
    ticket is charged, and different groups add. No acquire waits for memory.
    """
    def __init__(self, limit_bytes: int = 512 * 1024**2):
        self.limit_bytes = int(limit_bytes)
        self._frames: dict[str, tuple[FrameHandle, int]] = {}
        self._leases: dict[str, tuple[str, int, bool]] = {}
        self._scratch: dict[str, tuple[str, int]] = {}
        self._lock = threading.RLock()
        self.peak_bytes = 0

    def _pixel_bytes(self, frames=None) -> int:
        owners = {}
        for frame in (frame for frame, _ in self._frames.values()) if frames is None else frames:
            owner = frame.pixels
            while isinstance(owner, np.ndarray) and owner.base is not None:
                owner = owner.base
            while isinstance(owner, memoryview):
                owner = owner.obj
            size = owner.nbytes if isinstance(owner, np.ndarray) else len(owner) if isinstance(owner, bytes) else frame.nbytes
            owners[id(owner)] = max(owners.get(id(owner), 0), size)
        return sum(owners.values())

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._pixel_bytes() + max(
                (reserve for _, reserve, _ in self._leases.values()), default=0) + self._scratch_bytes()

    def _scratch_bytes(self) -> int:
        groups = {}
        for group, size in self._scratch.values():
            groups[group] = max(groups.get(group, 0), size)
        return sum(groups.values())

    def acquire_with_scratch(self, frame: FrameHandle, *, group: str, scratch_bytes: int,
                             reserve_bytes: int = 0, keep_preview_slots: int = 0) -> tuple[str, str]:
        """Atomically admit one raw lease and a serial executor's scratch ticket."""
        if not isinstance(group, str) or not group:
            raise ValueError("编码工作区必须属于明确的串行任务组。")
        if type(scratch_bytes) is not int or scratch_bytes < 0:
            raise ValueError("编码工作区预留必须是非负整数字节数。")
        with self._lock:
            ticket = uuid.uuid4().hex
            self._scratch[ticket] = (group, scratch_bytes)
            try:
                lease = self.acquire(frame, reserve_bytes=reserve_bytes,
                                     keep_preview_slots=keep_preview_slots)
            except BaseException:
                self._scratch.pop(ticket, None)
                raise
            return lease, ticket

    def release_scratch(self, ticket: str) -> None:
        with self._lock:
            self._scratch.pop(ticket, None)

    def acquire(self, frame: FrameHandle, *, reserve_bytes: int = 0, preview: bool = False,
                keep_preview_slots: int = 0) -> str:
        with self._lock:
            item = self._frames.get(frame.frame_id)
            if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int) or reserve_bytes < 0:
                raise ValueError("采集缓冲预留必须是非负整数字节数。")
            if isinstance(keep_preview_slots, bool) or not isinstance(keep_preview_slots, int) or keep_preview_slots < 0:
                raise ValueError("预览流转预留必须是非负整数帧数。")
            reserved = max(reserve_bytes, max((r for _, r, _ in self._leases.values()), default=0))
            scratch = self._scratch_bytes()
            if keep_preview_slots:
                pinned = {key for key, _, is_preview in self._leases.values() if not is_preview}
                frames = [self._frames[key][0] for key in pinned]
                frames.append(frame)
                future = self._pixel_bytes(frames) + frame.nbytes * keep_preview_slots + reserved + scratch
                if future > self.limit_bytes:
                    raise FrameBudgetExceeded("图像缓冲正在写入或计算，已暂停新增测量；请稍后重试，当前草稿和视场记录保留。")
            used = self._pixel_bytes()
            if item is not None and item[0] is not frame:
                raise ValueError("来源重复使用了同一帧编号。")
            frames = [value for value, _ in self._frames.values()]
            frames.append(frame)
            required = self._pixel_bytes(frames) - used
            # reserve_bytes is shared source-ring headroom, not per-consumer
            # scratch. The largest live reservation survives smaller leases.
            if used + required + reserved + scratch > self.limit_bytes:
                raise FrameBudgetExceeded("图像缓冲已满，正在等待计算或原图写入；实时画面继续更新。")
            key = uuid.uuid4().hex
            self._frames[frame.frame_id] = (frame, 1 if item is None else item[1] + 1)
            try:
                self._leases[key] = (frame.frame_id, reserve_bytes, preview)
            except BaseException:
                if item is None:
                    del self._frames[frame.frame_id]
                else:
                    self._frames[frame.frame_id] = item
                raise
            self.peak_bytes = max(self.peak_bytes, used + required + reserved + scratch)
            return key

    def replace(self, lease: str | None, frame: FrameHandle, *, reserve_bytes: int = 0, preview: bool = False) -> str:
        """Replace a preview reference atomically; a failed swap keeps the old lease."""
        with self._lock:
            previous = self._leases.get(lease)
            previous_frame = self._frames[previous[0]][0] if previous else None
            if previous:
                self.release(lease)
            try:
                return self.acquire(frame, reserve_bytes=reserve_bytes, preview=preview)
            except Exception:
                if previous:
                    restored = self.acquire(previous_frame, reserve_bytes=previous[1], preview=previous[2])
                    self._leases[lease] = self._leases.pop(restored)
                raise

    def release(self, lease: str) -> None:
        with self._lock:
            item = self._leases.pop(lease, None)
            if item is None:
                return
            key = item[0]
            frame, count = self._frames[key]
            if count == 1:
                del self._frames[key]
            else:
                self._frames[key] = (frame, count - 1)


def frame_from_qimage(image, **kwargs) -> FrameHandle:
    """One owned copy at the Qt/source boundary; native compute borrows it."""
    from PySide6.QtGui import QImage
    converted = image.convertToFormat(QImage.Format.Format_ARGB32)
    if converted.isNull():
        raise ValueError("采集图像为空。")
    raw = np.frombuffer(converted.constBits(), dtype=np.uint8).reshape(
        converted.height(), converted.bytesPerLine())
    pixels = raw[:, :converted.width() * 4].reshape(converted.height(), converted.width(), 4).copy()
    return FrameHandle.snapshot(pixels, transfer_ownership=True, **kwargs)
