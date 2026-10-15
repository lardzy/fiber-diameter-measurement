"""Bounded, persistent raw-frame transport for the verified Microview API.

Only this spawned process opens the SDK device. Frames travel through one shared
memory slot with an acknowledgement; neither screenshots nor image files are
used. A blocked SDK call can be terminated without taking down the GUI.
"""
from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import time

import numpy as np

MAX_FRAME_BYTES = 128 * 1024 * 1024


def _microview_buffer_view(buffer, info):
    from fdm.services.capture import _microview_normalize_pixel_format, _microview_bytes_per_pixel
    width, height, length = int(info.Width), int(info.Heigth), int(info.Length)
    pixel_format = _microview_normalize_pixel_format(int(info.nColor))
    bpp = _microview_bytes_per_pixel(pixel_format)
    if not bpp or width <= 0 or height <= 0 or length <= 0 or length > MAX_FRAME_BYTES:
        raise ValueError("Microview 帧尺寸或像素格式无效。")
    if length > len(buffer) or width * height * bpp > length:
        raise ValueError("Microview 帧缓冲长度与元数据不一致。")
    active = width * bpp
    expected = (width + max(0, int(info.SkipPixel))) * bpp
    if expected >= active and expected * height <= length:
        stride = expected
    elif length % height == 0 and length // height >= active:
        stride = length // height
    else:
        stride = active
    data = np.frombuffer(buffer, np.uint8, count=length)
    rows = np.lib.stride_tricks.as_strided(data, shape=(height, active), strides=(stride, 1), writeable=False)
    return rows, pixel_format, width, height, bpp


def copy_microview_to_shared(buffer, info, destination) -> tuple[int, ...]:
    """Copy raw SDK pixels directly into the shared slot with row-sized scratch.

    No full decoded frame exists alongside the SDK allocation and shared slot.
    Formats and stride semantics are the existing verified capture.py contract.
    """
    rows, pixel_format, width, height, bpp = _microview_buffer_view(buffer, info)
    shape = (height, width) if pixel_format == 0 else (height, width, bpp if pixel_format in (2, 3) else 3)
    image = np.ndarray(shape, np.uint8, buffer=destination)
    if pixel_format in (0, 2, 3):
        np.copyto(image, rows.reshape(shape))
        if pixel_format == 3:
            image[..., 3] = 255
        return shape
    green_bits = 6 if pixel_format == 5 else 5
    for y in range(height):
        words = np.ascontiguousarray(rows[y]).view("<u2").astype(np.uint32)
        blue, green, red = words & 31, (words >> 5) & ((1 << green_bits) - 1), (words >> (5 + green_bits)) & 31
        image[y, :, 0] = (blue << 3) | (blue >> 2)
        image[y, :, 1] = (green << (8 - green_bits)) | (green >> (2 * green_bits - 8))
        image[y, :, 2] = (red << 3) | (red >> 2)
    return shape


def decode_microview_buffer(buffer, info) -> np.ndarray:
    """Owned conversion utility; streaming writes directly to its shared slot."""
    rows, pixel_format, width, height, bpp = _microview_buffer_view(buffer, info)
    if pixel_format == 0:
        return np.array(rows, copy=True, order="C")
    if pixel_format in (2, 3):
        image = np.array(rows.reshape(height, width, bpp), copy=True, order="C")
        if bpp == 4:
            image[..., 3] = 255
        return image
    # Existing SDK declarations call these RGB565/RGB555; expand the little-
    # endian words directly so Qt GUI types are unnecessary in the transport.
    words = np.array(rows, copy=True).view("<u2").reshape(height, width).astype(np.uint32)
    green_bits = 6 if pixel_format == 5 else 5
    blue = words & 31
    green = (words >> 5) & ((1 << green_bits) - 1)
    red = (words >> (5 + green_bits)) & 31
    image = np.empty((height, width, 3), np.uint8)
    image[..., 0] = (blue << 3) | (blue >> 2)
    image[..., 1] = (green << (8 - green_bits)) | (green >> (2 * green_bits - 8))
    image[..., 2] = (red << 3) | (red >> 2)
    return image


def microview_worker(connection, capacity: int, device_index: int, fps: float) -> None:
    """Spawn entry: one SDK handle, reused allocation, one unacknowledged frame."""
    from fdm.services.capture import MicroviewCaptureBackend, _microview_open_error_message
    memory = None
    backend = MicroviewCaptureBackend()
    handle = None
    try:
        dll = backend._ensure_library()
        if dll is None:
            raise RuntimeError("未找到 Microview SDK DLL。")
        raw = dll.MV_OpenDevice(int(device_index), True)
        if not raw:
            raise RuntimeError(_microview_open_error_message(dll, backend._dll_root))
        handle = int(raw)
        board = backend._get_board_type(raw)
        backend._configure_capture_defaults(dll, raw, board)
        if int(dll.MV_OperateDevice(handle, backend._MV_RUN)) == backend._MV_ERROR:
            raise RuntimeError("Microview 内存采样启动失败。")
        connection.send({"type": "started", "backend": "microview_memory", "board_type": board})
        buffer = None
        sequence = 0
        while True:
            if connection.poll():
                command = connection.recv()
                if command.get("command") == "stop":
                    break
            started = time.perf_counter()
            info = backend._MVImageInfo()
            if not dll.MV_SetDeviceParameter(handle, backend._PARAM_SET_GARBIMAGEINFO, ctypes.addressof(info)):
                raise RuntimeError("Microview SDK 未返回可验证的图像缓冲信息。")
            length = int(info.Length)
            if length <= 0 or length > capacity or length > MAX_FRAME_BYTES:
                raise RuntimeError("Microview 图像缓冲超出已分配内存预算。")
            if buffer is None or len(buffer) != length:
                # Release the previous allocation before growing or shrinking;
                # resolution changes must not momentarily retain two raw slots.
                buffer = None
                buffer = ctypes.create_string_buffer(length)
            returned = dll.MV_CaptureSingle(handle, False, ctypes.addressof(buffer), len(buffer), ctypes.byref(info))
            if not returned:
                raise RuntimeError("Microview MV_CaptureSingle 未确认返回帧。")
            # The verified preallocation contract must be honoured. Never read
            # an unrelated SDK-owned address with an inferred lifetime/length.
            if int(returned) != ctypes.addressof(buffer):
                raise RuntimeError("Microview SDK 未使用已验证的预分配缓冲区，停止采样。")
            rows, pixel_format, width, height, bpp = _microview_buffer_view(buffer, info)
            del rows
            channels = 1 if pixel_format == 0 else bpp if pixel_format in (2, 3) else 3
            image_bytes = width * height * channels
            if image_bytes > capacity:
                raise RuntimeError("解码后的 Microview 图像超出内存预算。")
            if len(buffer) + 4 * image_bytes > 512 * 1024 * 1024:
                raise RuntimeError("Microview 采集、共享帧与显示工作区合计超过 512 MiB 预算。")
            if memory is None or memory.size != image_bytes:
                old_memory = memory
                if old_memory is not None:
                    connection.send({"type": "release_buffer"})
                    if not connection.poll(5):
                        raise RuntimeError("共享采集槽切换等待超时。")
                    command = connection.recv()
                    if command.get("command") == "stop":
                        break
                    if command != {"command": "buffer_released"}:
                        raise RuntimeError("共享采集槽切换确认无效。")
                    old_memory.close()
                    old_memory.unlink()
                    memory = None
                memory = shared_memory.SharedMemory(create=True, size=image_bytes)
                connection.send({"type": "buffer", "name": memory.name, "capacity": memory.size})
            shape = copy_microview_to_shared(buffer, info, memory.buf)
            sequence += 1
            metadata = {"type": "frame", "sequence": sequence, "received_at": time.perf_counter(),
                        "shape": shape, "nbytes": image_bytes,
                        "acquisition_reserved_bytes": memory.size + len(buffer),
                        "shared_slot_bytes": memory.size, "sdk_raw_bytes": len(buffer),
                        "decoded_frame_bytes": 0, "requested_fps_limit": fps}
            connection.send(metadata)
            # Retain this slot until its consumer owns a copy; identical pixels
            # still carry distinct successful SDK sample sequence numbers.
            if not connection.poll(5):
                raise RuntimeError("实时采集接收端未确认帧，已停止设备。")
            acknowledgement = connection.recv()
            if acknowledgement.get("command") == "stop":
                break
            if acknowledgement != {"command": "ack", "sequence": sequence}:
                raise RuntimeError("实时采集共享帧确认序号不匹配。")
            remaining = max(0.0, 1.0 / fps - (time.perf_counter() - started))
            if connection.poll(remaining):
                if connection.recv().get("command") == "stop":
                    break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"type": "error", "message": str(exc)})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if handle is not None and backend._dll is not None:
            backend._release_runtime_state(backend._dll, handle)
        backend._dispose_loaded_libraries()
        if memory is not None:
            memory.close()
            memory.unlink()
        connection.close()


class MicroviewMemoryClient:
    """The source's background thread owns start/poll/close; close is bounded."""
    def __init__(self, device_index: int, *, capacity: int = MAX_FRAME_BYTES, fps: float = 30):
        if not 1 <= capacity <= MAX_FRAME_BYTES:
            raise ValueError("共享图像内存预算无效。")
        self.device_index = int(device_index)
        self.capacity = int(capacity)
        self.fps = min(30.0, max(1.0, float(fps)))
        self._memory = None
        self._connection = None
        self._process = None
        self._deadline = 0.0
        self._sequence = 0
        self._started_metadata = None
        self._worker_target = microview_worker
        self._worker_arguments = (self.capacity, self.device_index, self.fps)
        self._worker_name = "fdm-microview-live"
        self._timeout_message = "Microview SDK 采样超时，已停止测量来源。"
        self._source_label = "Microview 内存采集"

    def start(self) -> None:
        if self._process is not None:
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=self._worker_target,
                                  args=(child, *self._worker_arguments),
                                  name=self._worker_name, daemon=True)
        self._connection, self._process = parent, process
        try:
            process.start()
            self._sequence = 0
            self._started_metadata = None
            self._deadline = time.perf_counter() + 8
        except BaseException:
            self.close()
            raise
        finally:
            child.close()

    def poll(self, timeout: float = .1):
        if self._connection is None or self._process is None:
            raise RuntimeError(f"{self._source_label}尚未启动。")
        if not self._connection.poll(timeout):
            if not self._process.is_alive():
                raise RuntimeError(f"{self._source_label}进程已退出。")
            if time.perf_counter() > self._deadline:
                raise TimeoutError(self._timeout_message)
            return None
        message = self._connection.recv()
        if message.get("type") == "error":
            raise RuntimeError(message.get("message", f"{self._source_label}失败。"))
        self._deadline = time.perf_counter() + 3
        if message.get("type") == "started":
            self._started_metadata = message
            return None
        if message.get("type") == "release_buffer":
            if self._memory is not None:
                self._memory.close()
                self._memory = None
            self._connection.send({"command": "buffer_released"})
            return None
        if message.get("type") == "buffer":
            capacity = int(message["capacity"])
            if not 1 <= capacity <= self.capacity:
                raise RuntimeError("Microview 共享槽大小无效。")
            memory = shared_memory.SharedMemory(name=message["name"])
            # Windows may report a page-rounded mapping length when attaching
            # by name. The advertised valid prefix is still exactly capacity.
            if memory.size < capacity:
                memory.close()
                raise RuntimeError("Microview 共享槽大小与协议不一致。")
            previous, self._memory = self._memory, memory
            if previous is not None:
                previous.close()
            return None
        if message.get("type") != "frame":
            raise RuntimeError("Microview 采集协议消息无效。")
        shape = tuple(int(value) for value in message["shape"])
        if len(shape) not in (2, 3) or any(value <= 0 for value in shape) or (len(shape) == 3 and shape[2] not in (3, 4)):
            raise RuntimeError("Microview 帧形状无效。")
        size = math.prod(shape)
        if self._memory is None or size != message["nbytes"] or size > self.capacity or size > self._memory.size:
            raise RuntimeError("Microview 帧长度无效。")
        sequence = int(message["sequence"])
        if sequence <= self._sequence:
            raise RuntimeError("Microview 帧序号未前进。")
        pixels = np.ndarray(shape, dtype=np.uint8, buffer=self._memory.buf).copy()
        self._connection.send({"command": "ack", "sequence": sequence})
        self._sequence = sequence
        return {**message, "pixels": pixels}

    def close(self) -> None:
        process, connection, memory = self._process, self._connection, self._memory
        if connection is not None:
            try:
                connection.send({"command": "stop"})
            except (EOFError, BrokenPipeError, OSError):
                pass
        if process is not None and process.pid is not None:
            process.join(.5)
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            if process.is_alive():
                raise RuntimeError(f"{self._source_label}进程未能确认退出。")
        if process is not None:
            process.close()
        # Keep ownership if even kill could not confirm exit: forgetting the
        # handles would make a later cleanup attempt impossible. Only release
        # shared memory after the writer is known to have stopped.
        self._process = self._connection = self._memory = None
        if connection is not None:
            connection.close()
        if memory is not None:
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                pass  # the normally exiting worker already unlinked its slot
