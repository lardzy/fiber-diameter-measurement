"""Bounded PrintWindow-only compatibility capture in a persistent process.

Used only when WGC is unavailable at startup. No desktop BitBlt fallback is
allowed; a hung target can be terminated without blocking the GUI or worker.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from multiprocessing import shared_memory
import time

import numpy as np

from .capture_helper import MAX_FRAME_BYTES, MicroviewMemoryClient


class _GeometryChanging(RuntimeError):
    """A moving window has no consistent geometry snapshot yet."""


class _WindowIdentity:
    def __init__(self, root: int, child: int):
        self.root, self.child = root, child
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.dwm = ctypes.WinDLL("dwmapi", use_last_error=True)
        signatures = {
            "IsWindow": ([wintypes.HWND], wintypes.BOOL),
            "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
            "IsIconic": ([wintypes.HWND], wintypes.BOOL),
            "GetAncestor": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
            "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            "GetClassNameW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetDlgCtrlID": ([wintypes.HWND], ctypes.c_int),
            "GetClientRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            "ClientToScreen": ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.user, name)
            function.argtypes, function.restype = arguments, result
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        self.dwm.DwmGetWindowAttribute.restype = ctypes.c_long
        if hasattr(self.user, "SetThreadDpiAwarenessContext"):
            self.user.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            self.user.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
            self.user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        if hasattr(self.user, "GetDpiForWindow"):
            self.user.GetDpiForWindow.argtypes = [wintypes.HWND]
            self.user.GetDpiForWindow.restype = wintypes.UINT
        self.pinned = self.read()

    def _process(self, hwnd):
        pid = wintypes.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = self.kernel.OpenProcess(0x1000, False, pid.value)
        if not handle:
            raise RuntimeError("capture_process_identity_unavailable")
        try:
            creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not self.kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
                raise RuntimeError("capture_process_identity_unavailable")
            return pid.value, (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        finally:
            self.kernel.CloseHandle(handle)

    def read(self):
        if not self.user.IsWindow(self.root) or not self.user.IsWindow(self.child):
            raise RuntimeError("capture_window_closed")
        if self.user.GetAncestor(self.child, 2) != self.root:
            raise RuntimeError("capture_window_identity_changed")
        root_identity, child_identity = self._process(self.root), self._process(self.child)
        if root_identity != child_identity:
            raise RuntimeError("capture_child_process_mismatch")
        name = ctypes.create_unicode_buffer(256)
        self.user.GetClassNameW(self.child, name, 256)
        return {"pid": root_identity[0], "process_created": root_identity[1],
                "root_hwnd": self.root, "child_hwnd": self.child,
                "child_class": name.value, "child_control_id": self.user.GetDlgCtrlID(self.child)}

    def geometry(self, api):
        if self.read() != self.pinned:
            raise RuntimeError("capture_window_identity_changed")
        cloaked = wintypes.DWORD()
        self.dwm.DwmGetWindowAttribute(self.root, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        if not self.user.IsWindowVisible(self.root) or not self.user.IsWindowVisible(self.child) or self.user.IsIconic(self.root) or cloaked.value:
            raise RuntimeError("capture_window_unavailable")
        # GetWindowRect and ClientToScreen are separate calls. A move between
        # them must not be mistaken for a permanently clipped child window.
        for _attempt in range(3):
            rectangle = api.get_window_rect(self.child)
            client = wintypes.RECT()
            if not self.user.GetClientRect(self.child, ctypes.byref(client)):
                raise RuntimeError("capture_child_geometry_unavailable")
            origin = wintypes.POINT(client.left, client.top)
            if not self.user.ClientToScreen(self.child, ctypes.byref(origin)):
                raise RuntimeError("capture_child_geometry_unavailable")
            if api.get_window_rect(self.child) == rectangle:
                break
        else:
            raise _GeometryChanging("capture_geometry_changing")
        crop = (origin.x - rectangle.left, origin.y - rectangle.top, client.right - client.left, client.bottom - client.top)
        x, y, width, height = crop
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > rectangle.width or y + height > rectangle.height:
            raise RuntimeError("capture_child_is_clipped")
        dpi = self.user.GetDpiForWindow(self.root) if hasattr(self.user, "GetDpiForWindow") else 96
        return rectangle, crop, dpi


def print_window_worker(connection, capacity: int, root: int, child: int, fps: float) -> None:
    from fdm.platform.windows_screen_capture import (
        _CtypesScreenCaptureApi, _BitmapInfo, _BitmapInfoHeader, CapturedFrame,
        BI_RGB, DIB_RGB_COLORS,
    )

    class SharedBitmapApi(_CtypesScreenCaptureApi):
        """Reuse PrintWindow and GDI lifetime code; read GDI directly into SHM."""
        memory = None

        def _read_unselected_bitmap(self, compatible_dc, bitmap, *, width, height):
            payload = (ctypes.c_ubyte * (width * height * 4)).from_buffer(self.memory.buf)
            info = _BitmapInfo()
            info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
            info.bmiHeader.biWidth, info.bmiHeader.biHeight = width, -height
            info.bmiHeader.biPlanes, info.bmiHeader.biBitCount = 1, 32
            info.bmiHeader.biCompression = BI_RGB
            count = self._gdi32.GetDIBits(compatible_dc, bitmap, 0, height, payload, ctypes.byref(info), DIB_RGB_COLORS)
            if count != height:
                raise RuntimeError("PrintWindow 兼容读取未返回完整图像。")
            return CapturedFrame(width, height, width * 4, self.memory.buf)

    memory = None
    frame = full = flat = sample = None
    try:
        identity, api = _WindowIdentity(root, child), SharedBitmapApi()
        connection.send({"type": "started", "identity": identity.pinned})
        previous_geometry, sequence, generation = None, 0, 0
        while True:
            if connection.poll() and connection.recv().get("command") == "stop":
                break
            started = time.perf_counter()
            try:
                rectangle, crop, dpi = identity.geometry(api)
            except _GeometryChanging:
                continue
            required = rectangle.width * rectangle.height * 4
            if not 1 <= required <= capacity:
                raise RuntimeError("PrintWindow 兼容捕获超出共享内存预算。")
            if memory is None or memory.size != required:
                if memory is not None:
                    connection.send({"type": "release_buffer"})
                    if not connection.poll(5):
                        raise TimeoutError("PrintWindow 共享槽切换超时。")
                    command = connection.recv()
                    if command.get("command") == "stop":
                        break
                    if command != {"command": "buffer_released"}:
                        raise RuntimeError("PrintWindow 共享槽确认无效。")
                    api.memory = None
                    memory.close()
                    memory.unlink()
                    memory = None
                memory = shared_memory.SharedMemory(create=True, size=required)
                api.memory = memory
                connection.send({"type": "buffer", "name": memory.name, "capacity": required})
            # Only this call sends WM_PRINT. It may hang in a target process,
            # hence this entire helper is terminable, and has no desktop path.
            frame = api.print_window(child, rectangle, include_cursor=False)
            try:
                geometry_after = identity.geometry(api)
            except _GeometryChanging:
                frame = None
                continue
            if geometry_after != (rectangle, crop, dpi):
                frame = None
                continue
            x, y, width, height = crop
            full = np.ndarray((rectangle.height, rectangle.width, 4), np.uint8, buffer=memory.buf)
            flat = np.ndarray((height, width, 4), np.uint8, buffer=memory.buf)
            for row in range(height):
                np.copyto(flat[row], full[y + row, x:x + width])
            flat[..., 3] = 255
            # A black GDI readback commonly means unsupported GPU content. It
            # must never silently become a valid camera measurement frame.
            sample = flat[::max(1, height // 64), ::max(1, width // 64), :3]
            if int(sample.max()) <= 8:
                raise RuntimeError("PrintWindow 兼容捕获只得到黑帧，此窗口不支持兼容模式。")
            del sample, flat, full, frame
            shape = (height, width, 4)
            current_geometry = (crop, dpi, rectangle.width, rectangle.height)
            if previous_geometry is not None and current_geometry != previous_geometry:
                generation += 1
            previous_geometry = current_geometry
            sequence += 1
            connection.send({"type": "frame", "shape": shape, "nbytes": width * height * 4,
                "sequence": sequence, "generation": generation, "dpi": dpi, "crop": crop,
                "source_timestamp_100ns": time.perf_counter_ns() // 100,
                "acquisition_reserved_bytes": required * 2, "backend": "printwindow_compat",
                "received_at": time.perf_counter()})
            if not connection.poll(5):
                raise TimeoutError("PrintWindow 接收端未确认帧。")
            command = connection.recv()
            if command.get("command") == "stop":
                break
            if command != {"command": "ack", "sequence": sequence}:
                raise RuntimeError("PrintWindow 帧序号确认无效。")
            if connection.poll(max(0, 1 / fps - (time.perf_counter() - started))):
                if connection.recv().get("command") == "stop":
                    break
    except (BrokenPipeError, EOFError):
        pass
    except Exception as exc:
        try:
            connection.send({"type": "error", "message": str(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        frame = full = flat = sample = None
        if memory is not None:
            try:
                memory.close()
            except BufferError:
                pass
            memory.unlink()
        connection.close()


class PrintWindowCaptureClient(MicroviewMemoryClient):
    def __init__(self, root_hwnd: int, child_hwnd: int):
        super().__init__(0, capacity=MAX_FRAME_BYTES, fps=30)
        self._worker_target = print_window_worker
        self._worker_arguments = (self.capacity, int(root_hwnd), int(child_hwnd), self.fps)
        self._worker_name = "fdm-printwindow-compat"
        self._timeout_message = "PrintWindow 兼容捕获超时，已停止目标调用。"
        self._source_label = "PrintWindow 兼容捕获"
        self.start()

    def identity(self):
        while self._started_metadata is None:
            super().poll(.1)
        return dict(self._started_metadata["identity"])

    def poll(self, timeout_ms=100):
        return super().poll(max(0, min(1000, timeout_ms)) / 1000)
