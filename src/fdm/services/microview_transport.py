"""SDK-processed frames for the existing helper; no second device owner.

The C++ callback owns two preallocated slots. The helper copies the latest
frame into one acknowledged shared slot. The parent decodes to an owned array
before acknowledging, so an SDK writer can never alter a measurement frame.
"""
from __future__ import annotations

import sys
import sysconfig
from multiprocessing import shared_memory
from types import SimpleNamespace

from fdm.live_experiment.capture_helper import MAX_FRAME_BYTES, decode_microview_buffer


MICROVIEW_BRIDGE_API_VERSION = 3
LATEST_MARKER_BYTES = 8


def _marker_offset(capacity):
    return (int(capacity) + LATEST_MARKER_BYTES - 1) & ~(LATEST_MARKER_BYTES - 1)


def bridge_build_matches(capture_info, kernel_info):
    """Separate extension interfaces share a build, not an API version."""
    return (capture_info.get("api_version") == MICROVIEW_BRIDGE_API_VERSION
            and capture_info.get("image_info_size") == 20
            and all(capture_info.get(key) == kernel_info.get(key)
                    for key in ("python_abi", "source_sha256")))


def load_bridge():
    from fdm._native import _microview, _source_root, native_source_sha256
    info = _microview.build_info()
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    if info.get("api_version") != MICROVIEW_BRIDGE_API_VERSION or info.get("image_info_size") != 20 or info.get("python_abi") != abi:
        raise RuntimeError("Microview 原生适配器接口或 Python ABI 不匹配，请重新构建。")
    root = _source_root()
    if root is not None and info.get("source_sha256") != native_source_sha256(root):
        raise RuntimeError("Microview 原生适配器已过期，请重新构建。")
    return _microview


class FramePublisher:
    """Used exclusively on the helper control thread, outside the SDK callback."""

    def __init__(self, backend, emit):
        self.backend, self.emit = backend, emit
        self.capacity = backend._frame_tap_capacity
        self.marker_offset = _marker_offset(self.capacity)
        self.memory = shared_memory.SharedMemory(create=True, size=self.marker_offset + LATEST_MARKER_BYTES)
        self.pending = None
        self.closed = False
        try:
            backend._frame_tap.bind_latest_marker(self.memory.buf[self.marker_offset:self.marker_offset + LATEST_MARKER_BYTES])
        except Exception:
            self.memory.close(); self.memory.unlink()
            raise
        emit({"type": "buffer", "name": self.memory.name, "capacity": self.capacity,
              "latest_marker_offset": self.marker_offset})

    def publish(self):
        if self.pending is not None:
            return False
        required = int(self.backend._frame_tap.diagnostics().get("required_capacity", 0))
        if required > self.capacity:
            if required > MAX_FRAME_BYTES:
                raise ValueError("Microview 图像超过共享帧容量限制。")
            marker_offset = _marker_offset(required)
            replacement = shared_memory.SharedMemory(create=True, size=marker_offset + LATEST_MARKER_BYTES)
            try:
                self.backend._frame_tap.reserve(required)
                self.backend._frame_tap.bind_latest_marker(
                    replacement.buf[marker_offset:marker_offset + LATEST_MARKER_BYTES])
            except Exception:
                replacement.close(); replacement.unlink()
                raise
            previous = self.memory
            self.memory, self.capacity = replacement, required
            self.marker_offset = marker_offset
            try:
                self.emit({"type": "buffer", "name": replacement.name, "capacity": required,
                           "latest_marker_offset": marker_offset})
            finally:
                previous.close(); previous.unlink()
        info = self.backend._frame_tap.poll_into(self.memory.buf)
        if info is None:
            return False
        self.pending = int(info["sequence"])
        self.emit({"type": "frame", **info,
                   "diagnostics": self.backend._frame_tap.diagnostics()})
        return True

    def acknowledge(self, sequence):
        if self.pending is None or int(sequence) != self.pending:
            raise RuntimeError("Microview 共享帧确认序号不匹配。")
        self.pending = None

    def close(self):
        if not self.closed:
            self.closed = True
            tap = getattr(self.backend, "_frame_tap", None)
            if tap is not None:
                tap.bind_latest_marker(None)
            self.memory.close()
            self.memory.unlink()


class FrameReceiver:
    """Only owned decoded arrays leave this object; no borrowed mapping views."""

    def __init__(self):
        self.memory = None
        self.capacity = 0
        self.sequence = 0
        self.superseded_frames = 0
        self._latest_marker = None
        self._read_latest_marker = None

    def attach(self, message):
        capacity = int(message["capacity"])
        if not 0 < capacity <= MAX_FRAME_BYTES:
            raise ValueError("Microview 共享帧容量无效。")
        memory = shared_memory.SharedMemory(name=str(message["name"]))
        marker_offset = message.get("latest_marker_offset")
        if marker_offset is not None:
            marker_offset = int(marker_offset)
        if (memory.size < capacity or (marker_offset is not None and
                (marker_offset != _marker_offset(capacity)
                 or memory.size < marker_offset + LATEST_MARKER_BYTES))):
            memory.close()
            raise ValueError("Microview 共享帧容量与协议不一致。")
        try:
            read_latest_marker = load_bridge().read_latest_marker if marker_offset is not None else None
        except Exception:
            memory.close()
            raise
        self.close()
        self.memory, self.capacity, self.sequence = memory, capacity, 0
        self.superseded_frames = 0
        self._read_latest_marker = read_latest_marker
        if marker_offset is not None:
            self._latest_marker = memory.buf[marker_offset:marker_offset + LATEST_MARKER_BYTES]

    def receive(self, message):
        sequence, length = int(message["sequence"]), int(message["length"])
        if self.memory is None or not 0 < length <= self.capacity or sequence <= self.sequence:
            raise ValueError("Microview 帧长度或序号无效。")
        # Acknowledgement protects the pending pixel slot, but it may be old
        # after a GUI stall. The callback updates this marker independently of
        # helper/GUI dispatch. Discard only when a newer valid frame exists;
        # low-rate cameras and identical live pixels remain valid indefinitely.
        # Check once before copying so a slow large-frame decode cannot starve.
        if (self._latest_marker is not None
                and self._read_latest_marker(self._latest_marker) > sequence):
            self.sequence = sequence
            self.superseded_frames += 1
            return None
        info = SimpleNamespace(Length=length, Width=int(message["width"]), Heigth=int(message["height"]),
                               nColor=int(message["color"]), SkipPixel=int(message["skip"]))
        pixels = decode_microview_buffer(self.memory.buf, info)
        self.sequence = sequence
        return pixels

    def close(self):
        if self._latest_marker is not None:
            self._latest_marker.release()
        self._latest_marker, self._read_latest_marker = None, None
        if self.memory is not None:
            self.memory.close()
        self.memory, self.capacity = None, 0
