from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from fdm.runtime_logging import append_runtime_log
from fdm.settings import project_runtime_root, runtime_directory

QCamera = None
QCameraDevice = object
QMediaCaptureSession = None
QMediaDevices = None
QVideoSink = None
_QT_MULTIMEDIA_IMPORT_ATTEMPTED = False
_QT_MULTIMEDIA_IMPORT_ERROR: Exception | None = None


@dataclass(slots=True)
class CaptureDevice:
    id: str
    name: str
    backend_key: str
    native_id: object
    available: bool = True
    detail: str = ""


class CaptureBackend:
    backend_key = "unknown"

    def list_devices(self) -> list[CaptureDevice]:
        return []

    def preview_kind(self, device: CaptureDevice) -> str:
        return "frame_stream"

    def preview_resolution(self, device: CaptureDevice) -> tuple[int, int] | None:
        return None

    def start_preview(
        self,
        device: CaptureDevice,
        *,
        preview_target: object | None = None,
        frame_callback: Callable[[QImage], None],
        error_callback: Callable[[str], None],
    ) -> None:
        raise NotImplementedError

    def stop_preview(self) -> None:
        raise NotImplementedError

    def update_preview_target(self, preview_target: object | None) -> None:
        return None

    def can_capture_still(self, device: CaptureDevice) -> bool:
        return False

    def capture_still_frame(self, device: CaptureDevice) -> QImage | None:
        return None

    def can_request_analysis_frame(self, device: CaptureDevice) -> bool:
        return False

    def request_analysis_frame(
        self,
        device: CaptureDevice,
        *,
        request_id: int,
        frame_callback: Callable[[int, QImage], None],
        error_callback: Callable[[int, str], None],
    ) -> None:
        raise RuntimeError("当前采集设备不支持分析帧采样。")

    def can_optimize_signal(self, device: CaptureDevice) -> bool:
        return False

    def optimize_signal(self, device: CaptureDevice) -> str:
        raise RuntimeError("当前采集设备不支持信号优化。")

    def active_warning(self) -> str:
        return ""

    def last_capture_diagnostics(self) -> str:
        return ""


def available_capture_backends() -> list[CaptureBackend]:
    return [
        MicroviewIsolatedBackend(),
        QtVideoCaptureBackend(),
    ]


class CaptureSessionManager(QObject):
    devicesChanged = Signal(object)
    previewStateChanged = Signal(bool)
    frameReady = Signal(object)
    analysisFrameReady = Signal(int, object)
    analysisFrameFailed = Signal(int, str)
    errorOccurred = Signal(str)
    _deliverFrame = Signal(int, object)
    _deliverError = Signal(int, str)
    _deliverAnalysisFrame = Signal(int, int, object)
    _deliverAnalysisError = Signal(int, int, str)

    def __init__(
        self,
        backends: list[CaptureBackend] | None = None,
        *,
        selected_device_id: str = "",
        refresh_on_init: bool = True,
    ) -> None:
        super().__init__()
        self._backends = backends or available_capture_backends()
        self._devices: list[CaptureDevice] = []
        self._selected_device_id = selected_device_id
        self._active_device_id = ""
        self._active_preview_kind = "frame_stream"
        self._active_preview_target: object | None = None
        self._active_backend: CaptureBackend | None = None
        self._last_frame: QImage | None = None
        self._preview_generation = 0
        self._device_refresh_warnings: list[str] = []
        self._deliverFrame.connect(self._on_frame_ready, Qt.ConnectionType.QueuedConnection)
        self._deliverError.connect(self._on_backend_error, Qt.ConnectionType.QueuedConnection)
        self._deliverAnalysisFrame.connect(self._on_analysis_frame_ready, Qt.ConnectionType.QueuedConnection)
        self._deliverAnalysisError.connect(self._on_analysis_error, Qt.ConnectionType.QueuedConnection)
        if refresh_on_init:
            self.refresh_devices()

    def devices(self) -> list[CaptureDevice]:
        return list(self._devices)

    def selected_device_id(self) -> str:
        return self._selected_device_id

    def selected_device(self) -> CaptureDevice | None:
        for device in self._devices:
            if device.id == self._selected_device_id:
                return device
        return self._devices[0] if self._devices else None

    def is_preview_active(self) -> bool:
        return bool(self._active_device_id)

    def last_frame(self) -> QImage | None:
        return self._last_frame.copy() if self._last_frame is not None else None

    def preview_kind(self) -> str:
        if self.is_preview_active():
            return self._active_preview_kind
        device = self.selected_device()
        backend = self._backend_for_device(device)
        if device is None or backend is None:
            return "frame_stream"
        return backend.preview_kind(device)

    def preview_resolution(self) -> tuple[int, int] | None:
        device = self.selected_device()
        backend = self._active_backend if self.is_preview_active() and self._active_backend is not None else self._backend_for_device(device)
        if device is None or backend is None:
            return None
        return backend.preview_resolution(device)

    def can_capture_still(self) -> bool:
        device = self.selected_device()
        backend = self._backend_for_device(device)
        if device is None or backend is None:
            return False
        return backend.can_capture_still(device)

    def can_request_analysis_frame(self) -> bool:
        if not self.is_preview_active() or self._active_backend is None:
            return False
        device = self.selected_device()
        if device is None:
            return False
        return self._active_backend.can_request_analysis_frame(device)

    def request_analysis_frame(self, request_id: int) -> bool:
        if not self.is_preview_active() or self._active_backend is None:
            return False
        device = self.selected_device()
        if device is None or not self._active_backend.can_request_analysis_frame(device):
            return False
        generation = self._preview_generation
        try:
            self._active_backend.request_analysis_frame(
                device,
                request_id=request_id,
                frame_callback=lambda req_id, image, _generation=generation: self._deliver_analysis_frame_threadsafe(
                    _generation, req_id, image
                ),
                error_callback=lambda req_id, message, _generation=generation: self._deliver_analysis_error_threadsafe(
                    _generation, req_id, message
                ),
            )
        except Exception as exc:
            self._deliver_analysis_error_threadsafe(generation, request_id, str(exc))
            return False
        return True

    def capture_still_frame(self) -> QImage | None:
        device = self.selected_device()
        backend = self._backend_for_device(device)
        if device is None or backend is None or not backend.can_capture_still(device):
            return None
        image = backend.capture_still_frame(device)
        if image is None or image.isNull():
            return None
        if backend.preview_kind(device) == "frame_stream":
            self._last_frame = image.copy()
        return image.copy()

    def can_optimize_signal(self) -> bool:
        device = self.selected_device()
        backend = self._backend_for_device(device)
        if device is None or backend is None:
            return False
        return backend.can_optimize_signal(device)

    def optimize_signal(self) -> str:
        device = self.selected_device()
        backend = self._backend_for_device(device)
        if device is None or backend is None:
            raise RuntimeError("当前没有可优化的采集设备。")
        return backend.optimize_signal(device)

    def active_warning(self) -> str:
        if self._active_backend is None:
            return ""
        return self._active_backend.active_warning()

    def capture_diagnostics(self) -> str:
        device = self.selected_device()
        backend = self._active_backend if self.is_preview_active() and self._active_backend is not None else self._backend_for_device(device)
        if backend is None:
            return ""
        return backend.last_capture_diagnostics()

    def device_refresh_warnings(self) -> list[str]:
        return list(self._device_refresh_warnings)

    def refresh_devices(self) -> list[CaptureDevice]:
        devices: list[CaptureDevice] = []
        warnings: list[str] = []
        for backend in self._backends:
            try:
                devices.extend(list(backend.list_devices() or []))
            except Exception as exc:
                warnings.append(_format_backend_error(backend, exc))
        self._devices = devices
        self._device_refresh_warnings = warnings
        if not any(device.id == self._selected_device_id for device in devices):
            self._selected_device_id = devices[0].id if devices else ""
        self.devicesChanged.emit(list(self._devices))
        return self.devices()

    def set_selected_device(self, device_id: str) -> bool:
        if not any(device.id == device_id for device in self._devices):
            return False
        restart_preview = self.is_preview_active()
        preview_target = self._active_preview_target
        if restart_preview:
            self.stop_preview()
        self._selected_device_id = device_id
        if restart_preview:
            self.start_preview(preview_target=preview_target)
        return True

    def start_preview(self, *, preview_target: object | None = None) -> bool:
        if not self._devices:
            self.refresh_devices()
        device = self.selected_device()
        if device is None:
            self.errorOccurred.emit("当前没有可用的采集设备。")
            return False
        if self.is_preview_active():
            self.stop_preview()
        backend = self._backend_for_device(device)
        if backend is None:
            self.errorOccurred.emit("未找到对应的采集后端。")
            return False
        self._preview_generation += 1
        generation = self._preview_generation
        self._last_frame = None
        self._active_backend = backend
        self._active_device_id = device.id
        self._active_preview_kind = backend.preview_kind(device)
        self._active_preview_target = preview_target
        try:
            backend.start_preview(
                device,
                preview_target=preview_target,
                frame_callback=lambda image, _generation=generation: self._deliver_frame_threadsafe(_generation, image),
                error_callback=lambda message, _generation=generation: self._deliver_error_threadsafe(_generation, message),
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._active_backend = None
            self._active_device_id = ""
            self._active_preview_kind = "frame_stream"
            self._active_preview_target = None
            self.errorOccurred.emit(str(exc))
            return False
        self.previewStateChanged.emit(True)
        return True

    def stop_preview(self) -> None:
        if not self.is_preview_active():
            self._last_frame = None
            return
        self._preview_generation += 1
        device = next((item for item in self._devices if item.id == self._active_device_id), None)
        backend = self._backend_for_device(device) if device is not None else None
        if backend is not None:
            try:
                backend.stop_preview()
            except Exception:
                pass
            self._recreate_backend_if_needed(backend)
        self._active_device_id = ""
        self._active_preview_kind = "frame_stream"
        self._active_preview_target = None
        self._active_backend = None
        self._last_frame = None
        self.previewStateChanged.emit(False)

    def update_preview_target(self, preview_target: object | None) -> None:
        if not self.is_preview_active() or self._active_backend is None:
            return
        self._active_preview_target = preview_target
        try:
            self._active_backend.update_preview_target(preview_target)
        except Exception as exc:
            self._deliver_error_threadsafe(self._preview_generation, str(exc))

    def _backend_for_device(self, device: CaptureDevice | None) -> CaptureBackend | None:
        if device is None:
            return None
        for backend in self._backends:
            if backend.backend_key == device.backend_key:
                return backend
        return None

    def _recreate_backend_if_needed(self, backend: CaptureBackend) -> None:
        if backend.backend_key != "microview":
            return
        for index, candidate in enumerate(self._backends):
            if candidate is not backend:
                continue
            backend_type = type(candidate)
            try:
                self._backends[index] = backend_type()
            except Exception:
                pass
            break

    @Slot(int, object)
    def _on_frame_ready(self, generation: int, image: QImage) -> None:
        if generation != self._preview_generation or not self.is_preview_active():
            return
        if image.isNull():
            return
        self._last_frame = image
        self.frameReady.emit(image)

    @Slot(int, str)
    def _on_backend_error(self, generation: int, message: str) -> None:
        if generation != self._preview_generation:
            return
        if self.is_preview_active():
            self.stop_preview()
        self.errorOccurred.emit(message)

    def _deliver_frame_threadsafe(self, generation: int, image: QImage) -> None:
        self._deliverFrame.emit(generation, image)

    def _deliver_error_threadsafe(self, generation: int, message: str) -> None:
        self._deliverError.emit(generation, message)

    @Slot(int, int, object)
    def _on_analysis_frame_ready(self, generation: int, request_id: int, image: QImage) -> None:
        if generation != self._preview_generation or not self.is_preview_active():
            return
        if image.isNull():
            return
        self.analysisFrameReady.emit(request_id, image)

    @Slot(int, int, str)
    def _on_analysis_error(self, generation: int, request_id: int, message: str) -> None:
        if generation != self._preview_generation or not self.is_preview_active():
            return
        self.analysisFrameFailed.emit(request_id, message)

    def _deliver_analysis_frame_threadsafe(self, generation: int, request_id: int, image: QImage) -> None:
        self._deliverAnalysisFrame.emit(generation, request_id, image)

    def _deliver_analysis_error_threadsafe(self, generation: int, request_id: int, message: str) -> None:
        self._deliverAnalysisError.emit(generation, request_id, message)


class QtVideoCaptureBackend(CaptureBackend):
    backend_key = "qt_multimedia"

    def __init__(self) -> None:
        self._camera: QCamera | None = None
        self._session: QMediaCaptureSession | None = None
        self._sink: QVideoSink | None = None
        self._last_frame: QImage | None = None
        self._frame_slot = None
        self._error_slot = None

    def list_devices(self) -> list[CaptureDevice]:
        _load_qt_multimedia()
        if QMediaDevices is None:
            return []
        devices: list[CaptureDevice] = []
        for camera_device in QMediaDevices.videoInputs():
            token = _qt_camera_token(camera_device)
            devices.append(
                CaptureDevice(
                    id=f"{self.backend_key}:{token}",
                    name=camera_device.description() or "USB 相机",
                    backend_key=self.backend_key,
                    native_id=camera_device,
                )
            )
        return devices

    def start_preview(
        self,
        device: CaptureDevice,
        *,
        preview_target: object | None = None,
        frame_callback: Callable[[QImage], None],
        error_callback: Callable[[str], None],
    ) -> None:
        _load_qt_multimedia()
        if QCamera is None or QMediaCaptureSession is None or QVideoSink is None:
            detail = _qt_multimedia_error_detail()
            if detail:
                raise RuntimeError(f"当前运行环境未启用 QtMultimedia，相机预览不可用。\n{detail}")
            raise RuntimeError("当前运行环境未启用 QtMultimedia，相机预览不可用。")
        camera_device = device.native_id
        if not isinstance(camera_device, QCameraDevice):
            raise RuntimeError("无效的 USB 相机设备。")
        self.stop_preview()
        self._camera = QCamera(camera_device)
        self._session = QMediaCaptureSession()
        self._sink = QVideoSink()
        self._session.setCamera(self._camera)
        self._session.setVideoSink(self._sink)
        self._last_frame = None
        self._frame_slot = lambda frame: self._on_video_frame(frame, frame_callback)
        self._sink.videoFrameChanged.connect(self._frame_slot)
        try:  # pragma: no branch - signal signature depends on Qt runtime
            self._error_slot = lambda _error, text: error_callback(text or "USB 相机预览失败")
            self._camera.errorOccurred.connect(self._error_slot)
        except Exception:
            self._error_slot = None
        self._camera.start()

    def stop_preview(self) -> None:
        if self._session is not None:
            try:
                self._session.setVideoSink(None)
            except Exception:
                pass
            try:
                self._session.setCamera(None)
            except Exception:
                pass
        if self._sink is not None and self._frame_slot is not None:
            try:
                self._sink.videoFrameChanged.disconnect(self._frame_slot)
            except Exception:
                pass
        if self._camera is not None and self._error_slot is not None:
            try:
                self._camera.errorOccurred.disconnect(self._error_slot)
            except Exception:
                pass
        if self._camera is not None:
            self._camera.stop()
            self._camera.deleteLater()
        if self._sink is not None:
            self._sink.deleteLater()
        if self._session is not None:
            try:
                self._session.deleteLater()
            except Exception:
                pass
        self._camera = None
        self._session = None
        self._sink = None
        self._last_frame = None
        self._frame_slot = None
        self._error_slot = None

    def can_capture_still(self, device: CaptureDevice) -> bool:
        return self._last_frame is not None and not self._last_frame.isNull()

    def capture_still_frame(self, device: CaptureDevice) -> QImage | None:
        if self._last_frame is None or self._last_frame.isNull():
            return None
        return self._last_frame.copy()

    def can_request_analysis_frame(self, device: CaptureDevice) -> bool:
        return self._last_frame is not None and not self._last_frame.isNull()

    def request_analysis_frame(
        self,
        device: CaptureDevice,
        *,
        request_id: int,
        frame_callback: Callable[[int, QImage], None],
        error_callback: Callable[[int, str], None],
    ) -> None:
        del device
        if self._last_frame is None or self._last_frame.isNull():
            error_callback(request_id, "当前预览尚未提供可用分析帧。")
            return
        frame_callback(request_id, self._last_frame.copy())

    def _on_video_frame(self, frame, frame_callback: Callable[[QImage], None]) -> None:
        image = _qt_video_frame_to_qimage(frame)
        if image.isNull():
            return
        self._last_frame = image.copy()
        frame_callback(image)


class MicroviewCaptureBackend(CaptureBackend):
    backend_key = "microview"
    _MV_RUN = 1
    _MV_STOP = 0
    _MV_ERROR = 4
    _PARAM_GET_BOARD_TYPE = 0
    _PARAM_SET_GARBIMAGEINFO = 2
    _PARAM_BUFFERTYPE = 4
    _PARAM_DISP_WIDTH = 11
    _PARAM_DISP_HEIGHT = 10
    _PARAM_DISP_PRESENCE = 6
    _PARAM_DISP_WHND = 7
    _PARAM_DISP_TOP = 8
    _PARAM_DISP_LEFT = 9
    _PARAM_RESTOPCAPTURE = 301
    _PARAM_ADJUST_LUMINANCE = 15
    _PARAM_ADJUST_SATURATION = 17
    _PARAM_ADJUST_HUE = 18
    _PARAM_ADJUST_CONTRAST = 19
    _PARAM_WORK_SKIP = 36
    _PARAM_WORK_FIELD = 40
    _PARAM_GRAB_HEIGHT = 62
    _PARAM_GRAB_WIDTH = 63
    _PARAM_GRAB_BITDESCRIBE = 66
    _BUFFER_SYSTEM_MEMORY_DX = 0
    _BUFFER_SYSTEM_MEMORY_GDI = 1
    _BUFFER_VIDEO_MEMORY = 2
    _SHOW_CLOSE = 0
    _SHOW_OPEN = 1
    _DEFAULT_VIDEO_LEVEL = 128
    _COLLECTION_FRAME = 0
    _INTERLUDE = 1
    _FORMAT_MONO8 = 0
    _FORMAT_RGB1555 = 1
    _FORMAT_RGB24 = 2
    _FORMAT_ARGB8888 = 3
    _FORMAT_RGB565 = 5
    _FORMAT_RGB5515 = 6
    _SIGNAL_ISINTERLACE = 2
    _SIGNAL_XSHIFT = 3
    _SIGNAL_YSHIFT = 4
    _SIGNAL_XSIZE = 5
    _SIGNAL_YSIZE = 6

    _LEVIN_M10 = 0x00006010
    _LEVIN_M20 = 0x00006020
    _LEVIN_RGB10 = 0x00009010
    _LEVIN_RGB20 = 0x00009020
    _LEVIN_VGA100 = 0x00009030
    _LEVIN_VGA170 = 0x00009040
    _FRAME_SKIP_BOARDS = {
        _LEVIN_M10,
        _LEVIN_M20,
        _LEVIN_RGB10,
        _LEVIN_RGB20,
        _LEVIN_VGA100,
        _LEVIN_VGA170,
    }
    _SIGNAL_OPTIMIZE_BOARDS = {
        _LEVIN_M20,
        _LEVIN_RGB20,
        _LEVIN_VGA100,
        _LEVIN_VGA170,
    }

    class _MVImageInfo(ctypes.Structure):
        _fields_ = [
            ("Length", ctypes.c_ulong),
            ("nColor", ctypes.c_ulong),
            ("Heigth", ctypes.c_ulong),
            ("Width", ctypes.c_ulong),
            ("SkipPixel", ctypes.c_ulong),
        ]

    def __init__(self) -> None:
        self._dll: ctypes.WinDLL | None = None
        self._dll_root: Path | None = None
        self._support_libraries: dict[str, ctypes.WinDLL] = {}
        self._dll_dirs: list[object] = []
        self._device_handle: int | None = None
        self._device_index: int | None = None
        self._board_type: int | None = None
        self._preview_target: object | None = None
        self._preview_buffer_type = self._BUFFER_SYSTEM_MEMORY_DX
        self._preview_resolution: tuple[int, int] | None = None
        self._active_warning = ""
        self._last_capture_diagnostics = ""
        self._frame_callback: Callable[[QImage], None] | None = None
        self._error_callback: Callable[[str], None] | None = None

    def preview_kind(self, device: CaptureDevice) -> str:
        return "native_embed"

    def preview_resolution(self, device: CaptureDevice) -> tuple[int, int] | None:
        if self._active_device_matches(device) and self._preview_resolution is not None:
            return self._preview_resolution
        return self._resolve_preview_resolution_for_device(device)

    def list_devices(self) -> list[CaptureDevice]:
        if sys.platform != "win32":
            return []
        try:
            dll = self._ensure_library()
        except OSError as exc:
            raise RuntimeError(f"Microview SDK 加载失败: {exc}") from exc
        if dll is None:
            return []
        try:
            device_count = int(dll.MV_GetDeviceNumber())
        except Exception as exc:
            raise RuntimeError(f"Microview 设备枚举失败: {exc}") from exc
        return [
            CaptureDevice(
                id=f"{self.backend_key}:{index}",
                name=f"Microview #{index + 1}",
                backend_key=self.backend_key,
                native_id=index,
            )
            for index in range(device_count)
        ]

    def start_preview(
        self,
        device: CaptureDevice,
        *,
        preview_target: object | None = None,
        frame_callback: Callable[[QImage], None],
        error_callback: Callable[[str], None],
    ) -> None:
        started_at = perf_counter()
        self.stop_preview()
        try:
            dll = self._ensure_library()
        except OSError as exc:
            raise RuntimeError(f"Microview SDK 加载失败: {exc}") from exc
        if dll is None:
            raise RuntimeError("未找到 Microview SDK DLL，无法启动实时预览。")
        if preview_target is None:
            raise RuntimeError("Microview 原生预览缺少目标窗口，无法启动预览。")
        self._frame_callback = frame_callback
        self._error_callback = error_callback
        self._preview_target = preview_target
        self._active_warning = ""
        raw_device_handle = dll.MV_OpenDevice(int(device.native_id), True)
        if not raw_device_handle:
            raise RuntimeError(_microview_open_error_message(dll, self._dll_root))
        self._device_handle = int(raw_device_handle)
        self._device_index = int(device.native_id)
        self._board_type = self._get_board_type(raw_device_handle)
        self._preview_resolution = self._get_capture_dimensions(raw_device_handle)
        self._configure_capture_defaults(dll, raw_device_handle, self._board_type)
        self._bind_preview_target(dll, raw_device_handle, preview_target)
        display_modes = (
            (self._BUFFER_SYSTEM_MEMORY_GDI, self._BUFFER_SYSTEM_MEMORY_DX)
            if _preview_target_prefers_gdi(preview_target)
            else (self._BUFFER_SYSTEM_MEMORY_DX, self._BUFFER_SYSTEM_MEMORY_GDI)
        )
        if not self._start_with_display_mode(dll, raw_device_handle, display_modes[0]):
            if self._start_with_display_mode(dll, raw_device_handle, display_modes[1]):
                self._active_warning = (
                    "GDI 预览初始化失败，已自动回退到 DirectX 预览。"
                    if display_modes[0] == self._BUFFER_SYSTEM_MEMORY_GDI
                    else "DirectX 预览初始化失败，已自动回退到 GDI 预览。"
                )
            else:
                self.stop_preview()
                raise RuntimeError("Microview 原生预览初始化失败，DirectX 和 GDI 模式均未成功。")
        self._log_perf(
            "Microview preview start",
            started_at,
            detail=(
                f"device={device.id}, board=0x{self._board_type or 0:08X}, "
                f"resolution={self._preview_resolution}, buffer={self._preview_buffer_type}"
            ),
        )

    def stop_preview(self) -> None:
        started_at = perf_counter()
        had_device_handle = self._device_handle is not None
        if self._dll is not None and self._device_handle:
            self._release_runtime_state(self._dll, self._device_handle)
        if self._dll is not None:
            self._dispose_loaded_libraries()
        self._device_handle = None
        self._device_index = None
        self._board_type = None
        self._preview_target = None
        self._preview_buffer_type = self._BUFFER_SYSTEM_MEMORY_DX
        self._preview_resolution = None
        self._active_warning = ""
        self._frame_callback = None
        self._error_callback = None
        if had_device_handle:
            self._log_perf("Microview preview stop", started_at)

    def update_preview_target(self, preview_target: object | None) -> None:
        if self._dll is None or not self._device_handle or preview_target is None:
            return
        self._preview_target = preview_target
        self._bind_preview_target(self._dll, self._device_handle, preview_target)

    def can_capture_still(self, device: CaptureDevice) -> bool:
        return True

    def capture_still_frame(self, device: CaptureDevice) -> QImage | None:
        self._last_capture_diagnostics = ""
        started_at = perf_counter()
        try:
            dll = self._ensure_library()
        except OSError as exc:
            raise RuntimeError(f"Microview SDK 加载失败: {exc}") from exc
        if dll is None:
            raise RuntimeError("未找到 Microview SDK DLL，无法抓拍图像。")
        temp_handle = None
        handle = None
        board_type = None
        temp_handle = dll.MV_OpenDevice(int(device.native_id), True)
        if not temp_handle:
            raise RuntimeError(_microview_open_error_message(dll, self._dll_root))
        handle = int(temp_handle)
        board_type = self._get_board_type(temp_handle)
        self._configure_capture_defaults(dll, temp_handle, board_type)
        result = int(dll.MV_OperateDevice(handle, self._MV_RUN))
        if result == self._MV_ERROR:
            self._release_runtime_state(dll, handle)
            temp_handle = None
            raise RuntimeError("Microview 独立抓拍启动失败。")
        if handle is None:
            return None
        try:
            image = self._capture_single_frame_image(handle=handle, process=False)
            if image.isNull():
                diagnostics = self._last_capture_diagnostics.strip()
                if diagnostics:
                    raise RuntimeError(f"Microview 抓拍失败。\n{diagnostics}")
                raise RuntimeError("Microview 抓拍失败，未返回有效图像。")
            self._log_perf(
                "Microview still capture",
                started_at,
                detail=(
                    f"device={device.id}, board=0x{board_type or 0:08X}, "
                    f"size={image.width()}x{image.height()}"
                ),
            )
            return image
        finally:
            if temp_handle:
                self._release_runtime_state(dll, handle)
            self._dispose_loaded_libraries()

    def can_request_analysis_frame(self, device: CaptureDevice) -> bool:
        del device
        return self._device_handle is not None and self._dll is not None

    def request_analysis_frame(
        self,
        device: CaptureDevice,
        *,
        request_id: int,
        frame_callback: Callable[[int, QImage], None],
        error_callback: Callable[[int, str], None],
    ) -> None:
        del device
        if self._device_handle is None or self._dll is None:
            error_callback(request_id, "Microview 预览尚未准备好分析帧。")
            return
        image = self._capture_single_frame_image(handle=self._device_handle, process=False)
        if image.isNull():
            diagnostics = self._last_capture_diagnostics.strip()
            error_callback(request_id, diagnostics or "Microview 分析帧抓取失败。")
            return
        frame_callback(request_id, image.copy())

    def can_optimize_signal(self, device: CaptureDevice) -> bool:
        try:
            board_type = self._resolve_board_type_for_device(device)
        except RuntimeError:
            return False
        return board_type in self._SIGNAL_OPTIMIZE_BOARDS

    def optimize_signal(self, device: CaptureDevice) -> str:
        try:
            dll = self._ensure_library()
        except OSError as exc:
            raise RuntimeError(f"Microview SDK 加载失败: {exc}") from exc
        if dll is None:
            raise RuntimeError("未找到 Microview SDK DLL，无法优化采集参数。")
        board_type = self._resolve_board_type_for_device(device)
        if board_type not in self._SIGNAL_OPTIMIZE_BOARDS:
            raise RuntimeError("当前板卡不支持自动优化采集参数。")
        raw_handle = dll.MV_OpenDevice(int(device.native_id), True)
        if not raw_handle:
            raise RuntimeError(_microview_open_error_message(dll, self._dll_root))
        handle = int(raw_handle)
        try:
            width = max(1, int(dll.MV_GetDeviceParameter(handle, self._PARAM_GRAB_WIDTH)))
            height = max(1, int(dll.MV_GetDeviceParameter(handle, self._PARAM_GRAB_HEIGHT)))
            if not bool(dll.MV_TestSignal(handle, width, height)):
                raise RuntimeError("Microview 自动帧测失败。")
            details = self._signal_param_summary(dll, handle)
            if not bool(dll.MV_SaveSignalParamToIni(handle)):
                raise RuntimeError("Microview 自动帧测已完成，但保存参数到 Ini 失败。")
            return f"已完成 Microview 采集参数优化并保存。\n{details}"
        finally:
            try:
                dll.MV_CloseDevice(handle)
            except Exception:
                pass

    def active_warning(self) -> str:
        return self._active_warning

    def last_capture_diagnostics(self) -> str:
        return self._last_capture_diagnostics

    def _ensure_library(self) -> ctypes.WinDLL | None:
        if self._dll is not None:
            return self._dll
        if sys.platform != "win32":
            return None
        dll_root = _resolve_microview_dll_root()
        if dll_root is None:
            return None
        self._dll_root = dll_root
        add_dir = getattr(os, "add_dll_directory", None)
        if callable(add_dir):
            self._dll_dirs.append(add_dir(str(dll_root)))
        self._preload_support_libraries(dll_root)
        dll_path = dll_root / "MVAPI.dll"
        loader = getattr(getattr(ctypes, "windll", None), "LoadLibrary", None)
        if callable(loader):
            dll = loader(str(dll_path))
        else:
            dll = ctypes.WinDLL(str(dll_path))
        dll.MV_GetDeviceNumber.restype = ctypes.c_ulong
        dll.MV_GetLastError.argtypes = [ctypes.c_bool]
        dll.MV_GetLastError.restype = ctypes.c_ulong
        dll.MV_OpenDevice.argtypes = [ctypes.c_ulong, ctypes.c_bool]
        dll.MV_OpenDevice.restype = ctypes.c_void_p
        dll.MV_CloseDevice.argtypes = [ctypes.c_void_p]
        dll.MV_OperateDevice.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.MV_OperateDevice.restype = ctypes.c_int
        dll.MV_SetDeviceParameter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        dll.MV_SetDeviceParameter.restype = ctypes.c_bool
        dll.MV_GetDeviceParameter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.MV_GetDeviceParameter.restype = ctypes.c_long
        dll.MV_CaptureSingle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(self._MVImageInfo),
        ]
        dll.MV_CaptureSingle.restype = ctypes.c_void_p
        dll.MV_TestSignal.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
        dll.MV_TestSignal.restype = ctypes.c_bool
        dll.MV_GetSignalParam.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        dll.MV_GetSignalParam.restype = ctypes.c_ulong
        dll.MV_SaveSignalParamToIni.argtypes = [ctypes.c_void_p]
        dll.MV_SaveSignalParamToIni.restype = ctypes.c_bool
        self._dll = dll
        return self._dll

    def _preload_support_libraries(self, dll_root: Path) -> None:
        loader = getattr(getattr(ctypes, "windll", None), "LoadLibrary", None)
        library_loader = loader if callable(loader) else ctypes.WinDLL
        for library_name in ("Function.dll", "MVBT.dll", "saa7130.dll", "mvavi.dll"):
            if library_name in self._support_libraries:
                continue
            library_path = _find_microview_support_library(library_name, dll_root)
            if library_path is None:
                continue
            try:
                self._support_libraries[library_name] = library_loader(str(library_path))
            except OSError:
                continue

    def _configure_capture_defaults(self, dll, device_handle, board_type: int | None) -> None:
        self._apply_frame_mode(dll, device_handle, board_type)
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_GRAB_BITDESCRIBE, self._FORMAT_ARGB8888)
        except Exception:
            pass
        for parameter in (
            self._PARAM_ADJUST_LUMINANCE,
            self._PARAM_ADJUST_CONTRAST,
            self._PARAM_ADJUST_HUE,
            self._PARAM_ADJUST_SATURATION,
        ):
            try:
                dll.MV_SetDeviceParameter(device_handle, parameter, self._DEFAULT_VIDEO_LEVEL)
            except Exception:
                pass

    def _apply_frame_mode(self, dll, device_handle, board_type: int | None) -> None:
        if board_type in self._FRAME_SKIP_BOARDS:
            try:
                dll.MV_SetDeviceParameter(device_handle, self._PARAM_WORK_SKIP, self._INTERLUDE)
            except Exception:
                pass
            return
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_WORK_FIELD, self._COLLECTION_FRAME)
        except Exception:
            pass

    def _bind_preview_target(self, dll, device_handle, preview_target: object) -> None:
        hwnd = _preview_target_handle(preview_target)
        width, height = _preview_target_dimensions(preview_target)
        if hwnd <= 0:
            raise RuntimeError("Microview 原生预览目标窗口无效。")
        dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_WHND, hwnd)
        dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_LEFT, 0)
        dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_TOP, 0)
        dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_WIDTH, width)
        dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_HEIGHT, height)
        dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_PRESENCE, self._SHOW_OPEN)

    def _start_with_display_mode(self, dll, device_handle, buffer_type: int) -> bool:
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_BUFFERTYPE, buffer_type)
        except Exception:
            return False
        try:
            result = int(dll.MV_OperateDevice(device_handle, self._MV_RUN))
        except Exception:
            return False
        if result == self._MV_ERROR:
            try:
                dll.MV_OperateDevice(device_handle, self._MV_STOP)
            except Exception:
                pass
            return False
        self._preview_buffer_type = buffer_type
        return True

    def _release_runtime_state(self, dll, handle: int | None) -> None:
        if not handle:
            return
        try:
            dll.MV_SetDeviceParameter(handle, self._PARAM_DISP_PRESENCE, self._SHOW_CLOSE)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(handle, self._PARAM_DISP_WHND, 0)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(handle, self._PARAM_DISP_LEFT, 0)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(handle, self._PARAM_DISP_TOP, 0)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(handle, self._PARAM_RESTOPCAPTURE, 0)
        except Exception:
            pass
        try:
            dll.MV_OperateDevice(handle, self._MV_STOP)
        except Exception:
            pass
        try:
            dll.MV_CloseDevice(handle)
        except Exception:
            pass

    def _dispose_loaded_libraries(self) -> None:
        if sys.platform != "win32":
            self._dll = None
            self._dll_root = None
            self._support_libraries.clear()
            self._dll_dirs.clear()
            return
        kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
        free_library = getattr(kernel32, "FreeLibrary", None)
        if callable(free_library):
            if self._dll is not None:
                try:
                    free_library(int(self._dll._handle))
                except Exception:
                    pass
            for library in list(self._support_libraries.values()):
                try:
                    free_library(int(library._handle))
                except Exception:
                    pass
        for dll_dir in self._dll_dirs:
            close = getattr(dll_dir, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._dll = None
        self._dll_root = None
        self._support_libraries.clear()
        self._dll_dirs.clear()

    def _log_perf(self, title: str, started_at: float, *, detail: str = "") -> None:
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        message = f"elapsed_ms={elapsed_ms:.2f}"
        if detail:
            message = f"{message}, {detail}"
        append_runtime_log(title, message)

    def _active_device_matches(self, device: CaptureDevice) -> bool:
        return self._device_handle is not None and self._device_index == int(device.native_id)

    def _resolve_board_type_for_device(self, device: CaptureDevice) -> int:
        if self._active_device_matches(device) and self._board_type is not None:
            return self._board_type
        try:
            dll = self._ensure_library()
        except OSError as exc:
            raise RuntimeError(f"Microview SDK 加载失败: {exc}") from exc
        if dll is None:
            raise RuntimeError("未找到 Microview SDK DLL，无法获取板卡信息。")
        raw_handle = dll.MV_OpenDevice(int(device.native_id), True)
        if not raw_handle:
            raise RuntimeError(_microview_open_error_message(dll, self._dll_root))
        try:
            return self._get_board_type(raw_handle)
        finally:
            try:
                dll.MV_CloseDevice(raw_handle)
            except Exception:
                pass

    def _resolve_preview_resolution_for_device(self, device: CaptureDevice) -> tuple[int, int] | None:
        if self._active_device_matches(device) and self._preview_resolution is not None:
            return self._preview_resolution
        try:
            dll = self._ensure_library()
        except OSError:
            return None
        if dll is None:
            return None
        raw_handle = dll.MV_OpenDevice(int(device.native_id), True)
        if not raw_handle:
            return None
        try:
            return self._get_capture_dimensions(raw_handle)
        finally:
            try:
                dll.MV_CloseDevice(raw_handle)
            except Exception:
                pass

    def _get_board_type(self, device_handle) -> int:
        if self._dll is None:
            return 0
        try:
            return int(self._dll.MV_GetDeviceParameter(device_handle, self._PARAM_GET_BOARD_TYPE))
        except Exception:
            return 0

    def _get_capture_dimensions(self, device_handle) -> tuple[int, int] | None:
        if self._dll is None:
            return None
        try:
            width = max(1, int(self._dll.MV_GetDeviceParameter(device_handle, self._PARAM_GRAB_WIDTH)))
            height = max(1, int(self._dll.MV_GetDeviceParameter(device_handle, self._PARAM_GRAB_HEIGHT)))
        except Exception:
            return None
        return width, height

    def _signal_param_summary(self, dll, device_handle) -> str:
        values: dict[str, int] = {}
        for label, signal in (
            ("interlace", self._SIGNAL_ISINTERLACE),
            ("xshift", self._SIGNAL_XSHIFT),
            ("yshift", self._SIGNAL_YSHIFT),
            ("xsize", self._SIGNAL_XSIZE),
            ("ysize", self._SIGNAL_YSIZE),
        ):
            float_val = ctypes.c_float()
            int_val = ctypes.c_ulong()
            try:
                dll.MV_GetSignalParam(device_handle, signal, ctypes.byref(float_val), ctypes.byref(int_val))
                values[label] = int(int_val.value)
            except Exception:
                continue
        if not values:
            return "已完成信号检测。"
        mode = "隔行" if values.get("interlace", 0) else "逐行"
        width = values.get("xsize", 0)
        height = values.get("ysize", 0)
        xshift = values.get("xshift", 0)
        yshift = values.get("yshift", 0)
        return f"检测结果: {mode}, size={width}x{height}, shift=({xshift}, {yshift})"

    def _capture_single_frame_image(self, *, handle: int | None = None, process: bool = True) -> QImage:
        if self._dll is None:
            return QImage()
        if handle is None:
            handle = self._device_handle
        if not handle:
            return QImage()
        diagnostics: list[str] = []
        flags: list[bool] = [process]
        if not process:
            flags.append(True)
        for attempt_index, process_flag in enumerate(flags, start=1):
            info = self._MVImageInfo()
            capture_buffer = None
            capture_ptr: int | None = None
            capture_length = 0
            set_info_ok = False
            try:
                set_info_ok = bool(self._dll.MV_SetDeviceParameter(
                    handle,
                    self._PARAM_SET_GARBIMAGEINFO,
                    ctypes.addressof(info),
                ))
            except Exception:
                set_info_ok = False
            if int(info.Length) > 0:
                capture_length = int(info.Length)
                capture_buffer = ctypes.create_string_buffer(capture_length)
                capture_ptr = ctypes.addressof(capture_buffer)
            sdk_error_before = self._microview_last_error_code()
            try:
                buffer_ptr = self._dll.MV_CaptureSingle(
                    handle,
                    process_flag,
                    capture_ptr,
                    capture_length,
                    ctypes.byref(info),
                )
            except Exception as exc:
                diagnostics.append(
                    "attempt="
                    f"{attempt_index}, process={process_flag}, set_info_ok={set_info_ok}, "
                    f"prealloc_len={capture_length}, error={exc.__class__.__name__}: {exc}"
                )
                continue
            resolved_ptr = int(buffer_ptr) if buffer_ptr else (capture_ptr or 0)
            diagnostics.append(
                "attempt="
                f"{attempt_index}, process={process_flag}, set_info_ok={set_info_ok}, "
                f"prealloc_len={capture_length}, sdk_ptr={int(buffer_ptr) if buffer_ptr else 0}, "
                f"resolved_ptr={resolved_ptr}, "
                f"info=(len={int(info.Length)}, w={int(info.Width)}, h={int(info.Heigth)}, color={int(info.nColor)}, skip={int(info.SkipPixel)}), "
                f"sdk_error_before={sdk_error_before}, sdk_error_after={self._microview_last_error_code()}"
            )
            if not resolved_ptr:
                continue
            try:
                image = _microview_buffer_to_qimage(resolved_ptr, info)
            except Exception as exc:
                diagnostics.append(
                    "attempt="
                    f"{attempt_index}, decode_error={exc.__class__.__name__}: {exc}"
                )
                continue
            if not image.isNull():
                diagnostics.append(
                    f"success=(w={image.width()}, h={image.height()}, process={process_flag})"
                )
                self._last_capture_diagnostics = "\n".join(diagnostics)
                return image
        self._last_capture_diagnostics = "\n".join(diagnostics)
        return QImage()

    def _microview_last_error_code(self) -> int:
        if self._dll is None:
            return 0
        try:
            return int(self._dll.MV_GetLastError(False))
        except Exception:
            return 0


class MicroviewIsolatedBackend(CaptureBackend):
    backend_key = "microview"

    def __init__(self) -> None:
        self._device_info: dict[str, dict[str, object]] = {}
        self._preview_process: QProcess | None = None
        self._preview_stdout_buffer = ""
        self._preview_stderr_buffer = ""
        self._preview_started = False
        self._active_device_id = ""
        self._active_resolution: tuple[int, int] | None = None
        self._active_warning = ""
        self._last_capture_diagnostics = ""
        self._pending_preview_error: str | None = None
        self._pending_snapshot_callbacks: dict[int, tuple[Callable[[int, QImage], None], Callable[[int, str], None]]] = {}

    def list_devices(self) -> list[CaptureDevice]:
        if sys.platform != "win32":
            return []
        response = self._run_helper_command("list")
        devices_payload = response.get("devices", [])
        if not isinstance(devices_payload, list):
            return []
        devices: list[CaptureDevice] = []
        self._device_info.clear()
        for item in devices_payload:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", len(devices)))
            device_id = f"{self.backend_key}:{index}"
            resolution_payload = item.get("resolution", [])
            resolution = None
            if isinstance(resolution_payload, list) and len(resolution_payload) == 2:
                resolution = (int(resolution_payload[0]), int(resolution_payload[1]))
            board_type = int(item.get("board_type", 0))
            devices.append(
                CaptureDevice(
                    id=device_id,
                    name=str(item.get("name") or f"Microview #{index + 1}"),
                    backend_key=self.backend_key,
                    native_id=index,
                )
            )
            self._device_info[device_id] = {
                "resolution": resolution,
                "board_type": board_type,
            }
        return devices

    def preview_kind(self, device: CaptureDevice) -> str:
        return "native_embed"

    def preview_resolution(self, device: CaptureDevice) -> tuple[int, int] | None:
        if self._active_device_id == device.id and self._active_resolution is not None:
            return self._active_resolution
        info = self._device_info.get(device.id, {})
        resolution = info.get("resolution")
        if isinstance(resolution, tuple) and len(resolution) == 2:
            return resolution
        return None

    def start_preview(
        self,
        device: CaptureDevice,
        *,
        preview_target: object | None = None,
        frame_callback: Callable[[QImage], None],
        error_callback: Callable[[str], None],
    ) -> None:
        del frame_callback
        if sys.platform != "win32":
            raise RuntimeError("Microview 实时预览仅支持 Windows。")
        if preview_target is None:
            raise RuntimeError("Microview 原生预览缺少目标窗口，无法启动预览。")
        self.stop_preview()
        program, arguments = _microview_helper_command(
            "preview",
            device_index=int(device.native_id),
            preview_target=preview_target,
        )
        process = QProcess()
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProgram(program)
        process.setArguments(arguments)
        self._preview_process = process
        self._preview_stdout_buffer = ""
        self._preview_stderr_buffer = ""
        self._preview_started = False
        self._active_device_id = device.id
        self._active_warning = ""
        self._active_resolution = None
        self._pending_preview_error = None
        process.readyReadStandardOutput.connect(lambda: self._drain_preview_stdout(error_callback))
        process.readyReadStandardError.connect(self._drain_preview_stderr)
        process.finished.connect(lambda exit_code, exit_status: self._on_preview_finished(exit_code, exit_status, error_callback))
        process.start()
        if not process.waitForStarted(5000):
            self._preview_process = None
            raise RuntimeError("Microview 预览 helper 进程启动失败。")
        deadline = perf_counter() + 5.0
        while not self._preview_started and process.state() == QProcess.ProcessState.Running and perf_counter() < deadline:
            process.waitForReadyRead(250)
            self._drain_preview_stdout(error_callback)
            self._drain_preview_stderr()
            if self._pending_preview_error:
                break
        if not self._preview_started:
            message = self._pending_preview_error or self._preview_stderr_buffer.strip() or "Microview 预览 helper 初始化失败。"
            self.stop_preview()
            raise RuntimeError(message)

    def stop_preview(self) -> None:
        process = self._preview_process
        if process is not None:
            if process.state() == QProcess.ProcessState.Running:
                try:
                    payload = json.dumps({"command": "stop"}, ensure_ascii=False).encode("utf-8") + b"\n"
                    process.write(payload)
                    process.waitForBytesWritten(500)
                    process.closeWriteChannel()
                    process.waitForFinished(2000)
                except Exception:
                    pass
            if process.state() == QProcess.ProcessState.Running:
                process.terminate()
                process.waitForFinished(1000)
            if process.state() == QProcess.ProcessState.Running:
                process.kill()
                process.waitForFinished(1000)
            process.deleteLater()
        self._preview_process = None
        self._preview_stdout_buffer = ""
        self._preview_stderr_buffer = ""
        self._preview_started = False
        self._active_device_id = ""
        self._active_resolution = None
        self._active_warning = ""
        self._pending_preview_error = None
        self._pending_snapshot_callbacks.clear()

    def update_preview_target(self, preview_target: object | None) -> None:
        process = self._preview_process
        if (
            preview_target is None
            or process is None
            or process.state() != QProcess.ProcessState.Running
            or not self._preview_started
        ):
            return None
        hwnd = _preview_target_handle(preview_target)
        width, height = _preview_target_dimensions(preview_target)
        if hwnd <= 0:
            return None
        self._send_preview_command(
            {
                "command": "update_target",
                "hwnd": int(hwnd),
                "width": int(width),
                "height": int(height),
            }
        )
        return None

    def can_capture_still(self, device: CaptureDevice) -> bool:
        del device
        return True

    def can_request_analysis_frame(self, device: CaptureDevice) -> bool:
        del device
        return self._preview_process is not None and self._preview_started

    def request_analysis_frame(
        self,
        device: CaptureDevice,
        *,
        request_id: int,
        frame_callback: Callable[[int, QImage], None],
        error_callback: Callable[[int, str], None],
    ) -> None:
        del device
        process = self._preview_process
        if process is None or process.state() != QProcess.ProcessState.Running or not self._preview_started:
            error_callback(request_id, "Microview 预览 helper 尚未准备好分析帧。")
            return
        self._pending_snapshot_callbacks[request_id] = (frame_callback, error_callback)
        self._send_preview_command(
            {
                "command": "snapshot",
                "request_id": int(request_id),
            }
        )

    def capture_still_frame(self, device: CaptureDevice) -> QImage | None:
        response = self._run_helper_command("capture", device_index=int(device.native_id), timeout_ms=15000)
        diagnostics = str(response.get("diagnostics", "")).strip()
        self._last_capture_diagnostics = diagnostics
        image_path = str(response.get("image_path", "")).strip()
        if not image_path:
            return None
        path = Path(image_path)
        image = QImage(str(path))
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        if image.isNull():
            return None
        return image

    def can_optimize_signal(self, device: CaptureDevice) -> bool:
        board_type = int(self._device_info.get(device.id, {}).get("board_type", 0))
        return board_type in MicroviewCaptureBackend._SIGNAL_OPTIMIZE_BOARDS

    def optimize_signal(self, device: CaptureDevice) -> str:
        response = self._run_helper_command("optimize", device_index=int(device.native_id), timeout_ms=15000)
        return str(response.get("message", "")).strip() or "已完成 Microview 采集参数优化。"

    def active_warning(self) -> str:
        return self._active_warning

    def last_capture_diagnostics(self) -> str:
        return self._last_capture_diagnostics

    def _run_helper_command(
        self,
        command: str,
        *,
        device_index: int | None = None,
        preview_target: object | None = None,
        timeout_ms: int = 8000,
    ) -> dict[str, object]:
        program, arguments = _microview_helper_command(
            command,
            device_index=device_index,
            preview_target=preview_target,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            [program, *arguments],
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_ms / 1000)),
            env=env,
        )
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        payload: dict[str, object] | None = None
        if stdout_lines:
            try:
                parsed = json.loads(stdout_lines[-1])
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = None
        if completed.returncode != 0:
            message = ""
            if isinstance(payload, dict):
                message = str(payload.get("message", "")).strip()
            if not message:
                message = completed.stderr.strip() or completed.stdout.strip() or f"Microview helper 执行失败: {command}"
            raise RuntimeError(message)
        if payload is None:
            raise RuntimeError(f"Microview helper 未返回有效响应: {command}")
        return payload

    def _drain_preview_stdout(self, error_callback: Callable[[str], None]) -> None:
        process = self._preview_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return
        self._preview_stdout_buffer += chunk
        while "\n" in self._preview_stdout_buffer:
            raw_line, self._preview_stdout_buffer = self._preview_stdout_buffer.split("\n", 1)
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            message_type = str(payload.get("type", "")).strip()
            if message_type == "started":
                resolution_payload = payload.get("resolution", [])
                if isinstance(resolution_payload, list) and len(resolution_payload) == 2:
                    self._active_resolution = (int(resolution_payload[0]), int(resolution_payload[1]))
                self._active_warning = str(payload.get("warning", "")).strip()
                self._preview_started = True
                continue
            if message_type == "snapshot":
                request_id = int(payload.get("request_id", 0))
                callbacks = self._pending_snapshot_callbacks.pop(request_id, None)
                if callbacks is None:
                    continue
                image_path = str(payload.get("image_path", "")).strip()
                path = Path(image_path)
                image = QImage(str(path))
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                if image.isNull():
                    callbacks[1](request_id, "Microview 分析帧读取失败。")
                else:
                    callbacks[0](request_id, image)
                continue
            if message_type == "snapshot_error":
                request_id = int(payload.get("request_id", 0))
                callbacks = self._pending_snapshot_callbacks.pop(request_id, None)
                if callbacks is not None:
                    callbacks[1](request_id, str(payload.get("message", "")).strip() or "Microview 分析帧抓取失败。")
                continue
            if message_type == "error":
                self._pending_preview_error = str(payload.get("message", "")).strip() or "Microview 预览 helper 出错。"
                if self._preview_started and self._pending_preview_error:
                    error_callback(self._pending_preview_error)

    def _drain_preview_stderr(self) -> None:
        process = self._preview_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        if chunk:
            self._preview_stderr_buffer += chunk

    def _on_preview_finished(self, exit_code: int, exit_status, error_callback: Callable[[str], None]) -> None:
        del exit_status
        process = self._preview_process
        if process is None:
            return
        self._drain_preview_stdout(error_callback)
        self._drain_preview_stderr()
        unexpected = self._preview_started and self._active_device_id and exit_code != 0
        self._preview_process = None
        self._preview_stdout_buffer = ""
        self._preview_started = False
        self._active_device_id = ""
        self._active_resolution = None
        self._active_warning = ""
        if unexpected:
            message = self._pending_preview_error or self._preview_stderr_buffer.strip() or "Microview 预览 helper 意外退出。"
            error_callback(message)
        self._preview_stderr_buffer = ""
        self._pending_preview_error = None
        pending = list(self._pending_snapshot_callbacks.items())
        self._pending_snapshot_callbacks.clear()
        for request_id, callbacks in pending:
            callbacks[1](request_id, "Microview 预览已结束，分析帧请求已取消。")

    def _send_preview_command(self, payload: dict[str, object]) -> None:
        process = self._preview_process
        if process is None or process.state() != QProcess.ProcessState.Running:
            return
        try:
            message = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
            process.write(message)
            process.waitForBytesWritten(500)
        except Exception:
            return


def _microview_helper_command(
    command: str,
    *,
    device_index: int | None = None,
    preview_target: object | None = None,
) -> tuple[str, list[str]]:
    arguments: list[str]
    if getattr(sys, "frozen", False):
        program = sys.executable
        arguments = ["--microview-helper", command]
    else:
        program = sys.executable
        arguments = ["-u", "-m", "fdm.microview_helper", command]
    if device_index is not None:
        arguments.extend(["--device-index", str(int(device_index))])
    if command == "preview":
        hwnd = _preview_target_handle(preview_target)
        width, height = _preview_target_dimensions(preview_target) if preview_target is not None else (640, 480)
        arguments.extend(
            [
                "--preview-hwnd",
                str(int(hwnd)),
                "--preview-width",
                str(int(width)),
                "--preview-height",
                str(int(height)),
            ]
        )
        if preview_target is not None and _preview_target_prefers_gdi(preview_target):
            arguments.append("--prefer-gdi")
    return program, arguments


def _qt_camera_token(device: QCameraDevice) -> str:
    try:
        raw_id = bytes(device.id())
        if raw_id:
            return raw_id.hex()
    except Exception:
        pass
    description = device.description() or "camera"
    return description.strip().replace(" ", "_")


def _qt_video_frame_to_qimage(frame) -> QImage:
    if frame is None or not getattr(frame, "isValid", lambda: False)():
        return QImage()
    image = frame.toImage()
    if image.isNull():
        return QImage()
    return image.copy()


def _load_qt_multimedia() -> None:
    global QCamera, QCameraDevice, QMediaCaptureSession, QMediaDevices, QVideoSink
    global _QT_MULTIMEDIA_IMPORT_ATTEMPTED, _QT_MULTIMEDIA_IMPORT_ERROR
    if _QT_MULTIMEDIA_IMPORT_ATTEMPTED:
        return
    _QT_MULTIMEDIA_IMPORT_ATTEMPTED = True
    try:
        from PySide6.QtMultimedia import QCamera as _QCamera
        from PySide6.QtMultimedia import QCameraDevice as _QCameraDevice
        from PySide6.QtMultimedia import QMediaCaptureSession as _QMediaCaptureSession
        from PySide6.QtMultimedia import QMediaDevices as _QMediaDevices
        from PySide6.QtMultimedia import QVideoSink as _QVideoSink
    except Exception as exc:  # pragma: no cover - depends on runtime Qt plugins
        _QT_MULTIMEDIA_IMPORT_ERROR = exc
        return
    QCamera = _QCamera
    QCameraDevice = _QCameraDevice
    QMediaCaptureSession = _QMediaCaptureSession
    QMediaDevices = _QMediaDevices
    QVideoSink = _QVideoSink


def _qt_multimedia_error_detail() -> str:
    if _QT_MULTIMEDIA_IMPORT_ERROR is None:
        return ""
    return str(_QT_MULTIMEDIA_IMPORT_ERROR).strip()


def _format_backend_error(backend: CaptureBackend, exc: Exception) -> str:
    name = {
        "microview": "Microview SDK",
        "qt_multimedia": "USB 相机",
    }.get(backend.backend_key, backend.backend_key)
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{name}: {detail}"


def _microview_open_error_message(dll, dll_root: Path | None = None) -> str:
    try:
        error_code = int(dll.MV_GetLastError(False))
    except Exception:
        error_code = 0
    root_hint = f"\n当前 SDK 目录: {dll_root}" if dll_root is not None else ""
    if error_code == 2:
        message = (
            "Microview 设备打开失败。SDK 错误码: 2\n"
            "这通常表示 Microview 运行环境未完整安装，或设备正被其它程序占用。\n"
            "请先关闭官方采集软件/演示程序，再确认驱动与运行库已按 SDK 说明安装。"
        )
        details = _microview_runtime_diagnostics()
        if details:
            return f"{message}\n{details}{root_hint}"
        return f"{message}{root_hint}"
    if error_code > 0:
        return f"Microview 设备打开失败。SDK 错误码: {error_code}{root_hint}"
    return f"Microview 设备打开失败。{root_hint}".rstrip()


def _microview_runtime_diagnostics() -> str:
    if sys.platform != "win32":
        return ""
    system_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
    missing: list[str] = []
    for candidate in (
        system_root / "System32" / "drivers" / "MVBT.sys",
        system_root / "System32" / "drivers" / "saa7130.sys",
    ):
        if not candidate.exists():
            missing.append(candidate.name)
    if missing:
        return f"未检测到驱动文件: {', '.join(missing)}"
    return "系统驱动文件存在，更可能是设备被其它程序占用或官方运行库未注册。"


def _resolve_microview_dll_root() -> Path | None:
    candidates = _microview_dll_candidates()
    for candidate in candidates:
        if (candidate / "MVAPI.dll").exists():
            return candidate
    return None


def _microview_dll_candidates() -> list[Path]:
    arch = "x64" if sys.maxsize > 2 ** 32 else "x86"
    candidates: list[Path] = []
    env_dir = os.environ.get("FDM_MICROVIEW_DLL_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    if sys.platform == "win32":
        system_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        if arch == "x64":
            candidates.extend(
                [
                    system_root / "SysWOW64",
                    system_root / "System32",
                    system_root,
                ]
            )
        else:
            candidates.extend(
                [
                    system_root / "System32",
                    system_root / "SysWOW64",
                    system_root,
                ]
            )
    candidates.extend(
        [
            runtime_directory() / "camera" / "microview" / arch,
            project_runtime_root() / "dll" / arch,
            project_runtime_root() / ".tmp" / "Microview-ref" / "MVFG" / "sdk" / "dll" / arch,
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _find_microview_support_library(library_name: str, dll_root: Path) -> Path | None:
    search_roots: list[Path] = [dll_root]
    if sys.platform == "win32":
        system_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        search_roots.extend([system_root, system_root / "System32", system_root / "SysWOW64"])
    for root in search_roots:
        candidate = root / library_name
        if candidate.exists():
            return candidate
    return None


def _preview_target_handle(preview_target: object) -> int:
    if hasattr(preview_target, "native_preview_handle"):
        try:
            return int(preview_target.native_preview_handle())  # type: ignore[attr-defined]
        except Exception:
            return 0
    if hasattr(preview_target, "winId"):
        try:
            return int(preview_target.winId())  # type: ignore[attr-defined]
        except Exception:
            return 0
    return 0


def _preview_target_dimensions(preview_target: object) -> tuple[int, int]:
    if hasattr(preview_target, "native_preview_size"):
        try:
            width, height = preview_target.native_preview_size()  # type: ignore[attr-defined]
            return max(1, int(width)), max(1, int(height))
        except Exception:
            return (640, 480)
    width = max(1, int(getattr(preview_target, "width", lambda: 640)()))
    height = max(1, int(getattr(preview_target, "height", lambda: 480)()))
    return width, height


def _preview_target_prefers_gdi(preview_target: object) -> bool:
    preference = getattr(preview_target, "prefer_gdi_preview", None)
    if not callable(preference):
        return False
    try:
        return bool(preference())
    except Exception:
        return False


def _microview_pixel_format_name(pixel_format: int) -> str:
    return {
        MicroviewCaptureBackend._FORMAT_MONO8: "MONO8",
        MicroviewCaptureBackend._FORMAT_RGB1555: "RGB1555",
        MicroviewCaptureBackend._FORMAT_RGB24: "RGB24",
        MicroviewCaptureBackend._FORMAT_ARGB8888: "ARGB8888",
        MicroviewCaptureBackend._FORMAT_RGB565: "RGB565",
        MicroviewCaptureBackend._FORMAT_RGB5515: "RGB5515",
        7: "YUV444",
        8: "YUV422",
        9: "YUV411",
    }.get(pixel_format, str(pixel_format))


def _microview_normalize_pixel_format(pixel_format: int) -> int:
    aliases = {
        8: MicroviewCaptureBackend._FORMAT_MONO8,
        15: MicroviewCaptureBackend._FORMAT_RGB1555,
        16: MicroviewCaptureBackend._FORMAT_RGB565,
        24: MicroviewCaptureBackend._FORMAT_RGB24,
        32: MicroviewCaptureBackend._FORMAT_ARGB8888,
    }
    return aliases.get(pixel_format, pixel_format)


def _microview_bytes_per_pixel(pixel_format: int) -> int | None:
    pixel_format = _microview_normalize_pixel_format(pixel_format)
    return {
        MicroviewCaptureBackend._FORMAT_MONO8: 1,
        MicroviewCaptureBackend._FORMAT_RGB1555: 2,
        MicroviewCaptureBackend._FORMAT_RGB24: 3,
        MicroviewCaptureBackend._FORMAT_ARGB8888: 4,
        MicroviewCaptureBackend._FORMAT_RGB565: 2,
        MicroviewCaptureBackend._FORMAT_RGB5515: 2,
    }.get(pixel_format)


def _microview_qimage_format(pixel_format: int):
    pixel_format = _microview_normalize_pixel_format(pixel_format)
    if pixel_format == MicroviewCaptureBackend._FORMAT_MONO8:
        return QImage.Format.Format_Grayscale8
    if pixel_format == MicroviewCaptureBackend._FORMAT_RGB24:
        return getattr(QImage.Format, "Format_BGR888", QImage.Format.Format_RGB888)
    if pixel_format == MicroviewCaptureBackend._FORMAT_ARGB8888:
        return getattr(QImage.Format, "Format_ARGB32", QImage.Format.Format_RGBA8888)
    if pixel_format == MicroviewCaptureBackend._FORMAT_RGB565:
        return getattr(QImage.Format, "Format_RGB16", None)
    if pixel_format in (
        MicroviewCaptureBackend._FORMAT_RGB1555,
        MicroviewCaptureBackend._FORMAT_RGB5515,
    ):
        return getattr(QImage.Format, "Format_RGB555", None)
    return None


def _microview_compact_rows(data: bytes, row_stride: int, active_stride: int, height: int) -> bytes:
    chunks: list[bytes] = []
    for row_index in range(height):
        row_start = row_index * row_stride
        row_end = row_start + active_stride
        chunks.append(data[row_start:row_end])
    return b"".join(chunks)


def _microview_force_opaque_alpha(data: bytes) -> bytes:
    if not data:
        return data
    mutable = bytearray(data)
    for index in range(3, len(mutable), 4):
        mutable[index] = 0xFF
    return bytes(mutable)


def _microview_buffer_to_qimage(buffer_ptr: int, info: MicroviewCaptureBackend._MVImageInfo) -> QImage:
    width = int(info.Width)
    height = int(info.Heigth)
    total_length = int(info.Length)
    raw_pixel_format = int(info.nColor)
    pixel_format = _microview_normalize_pixel_format(raw_pixel_format)
    skip_pixels = max(0, int(info.SkipPixel))
    if width <= 0 or height <= 0 or total_length <= 0:
        return QImage()
    bytes_per_pixel = _microview_bytes_per_pixel(pixel_format)
    if bytes_per_pixel is None:
        raise ValueError(
            "不支持的 Microview 像素格式: "
            f"{_microview_pixel_format_name(raw_pixel_format)} ({raw_pixel_format})"
        )
    image_format = _microview_qimage_format(pixel_format)
    if image_format is None:
        raise ValueError(
            "当前 Qt 运行环境不支持 Microview 像素格式: "
            f"{_microview_pixel_format_name(raw_pixel_format)}"
        )
    data = ctypes.string_at(buffer_ptr, total_length)
    active_stride = width * bytes_per_pixel
    minimum_total = active_stride * height
    if len(data) < minimum_total:
        raise ValueError(
            "Microview 图像缓冲区长度不足: "
            f"{len(data)} < {minimum_total} ({width}x{height}, format={_microview_pixel_format_name(raw_pixel_format)})"
        )

    row_stride_candidates: list[int] = []
    expected_stride = (width + skip_pixels) * bytes_per_pixel
    if expected_stride >= active_stride and expected_stride * height <= len(data):
        row_stride_candidates.append(expected_stride)
    if len(data) % height == 0:
        derived_stride = len(data) // height
        if derived_stride >= active_stride and derived_stride not in row_stride_candidates:
            row_stride_candidates.append(derived_stride)
    if active_stride not in row_stride_candidates and minimum_total <= len(data):
        row_stride_candidates.append(active_stride)
    if not row_stride_candidates:
        raise ValueError(
            "无法确定 Microview 图像步长: "
            f"length={len(data)}, width={width}, height={height}, "
            f"skip_pixels={skip_pixels}, format={_microview_pixel_format_name(raw_pixel_format)}"
        )

    row_stride = row_stride_candidates[0]
    if row_stride * height > len(data):
        raise ValueError(
            "Microview 图像步长超过缓冲区长度: "
            f"stride={row_stride}, height={height}, length={len(data)}"
        )
    compact_data = data if row_stride == active_stride else _microview_compact_rows(data, row_stride, active_stride, height)
    if pixel_format == MicroviewCaptureBackend._FORMAT_ARGB8888:
        compact_data = _microview_force_opaque_alpha(compact_data)
    image = QImage(compact_data, width, height, active_stride, image_format)
    if image.isNull():
        raise ValueError(
            "Qt 无法构建 Microview 图像: "
            f"{width}x{height}, stride={active_stride}, format={_microview_pixel_format_name(raw_pixel_format)}"
        )
    return image.copy()
