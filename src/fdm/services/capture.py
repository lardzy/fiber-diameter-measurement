from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import ctypes
import os
import sys

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage

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

    def start_preview(
        self,
        device: CaptureDevice,
        *,
        frame_callback: Callable[[QImage], None],
        error_callback: Callable[[str], None],
    ) -> None:
        raise NotImplementedError

    def stop_preview(self) -> None:
        raise NotImplementedError


def available_capture_backends() -> list[CaptureBackend]:
    return [
        MicroviewCaptureBackend(),
        QtVideoCaptureBackend(),
    ]


class CaptureSessionManager(QObject):
    devicesChanged = Signal(object)
    previewStateChanged = Signal(bool)
    frameReady = Signal(object)
    errorOccurred = Signal(str)

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
        self._last_frame: QImage | None = None
        self._device_refresh_warnings: list[str] = []
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
        if restart_preview:
            self.stop_preview()
        self._selected_device_id = device_id
        if restart_preview:
            self.start_preview()
        return True

    def start_preview(self) -> bool:
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
        self._last_frame = None
        try:
            backend.start_preview(
                device,
                frame_callback=self._on_frame_ready,
                error_callback=self._on_backend_error,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.errorOccurred.emit(str(exc))
            return False
        self._active_device_id = device.id
        self.previewStateChanged.emit(True)
        return True

    def stop_preview(self) -> None:
        if not self.is_preview_active():
            return
        device = next((item for item in self._devices if item.id == self._active_device_id), None)
        backend = self._backend_for_device(device) if device is not None else None
        if backend is not None:
            try:
                backend.stop_preview()
            except Exception:
                pass
        self._active_device_id = ""
        self.previewStateChanged.emit(False)

    def _backend_for_device(self, device: CaptureDevice | None) -> CaptureBackend | None:
        if device is None:
            return None
        for backend in self._backends:
            if backend.backend_key == device.backend_key:
                return backend
        return None

    def _on_frame_ready(self, image: QImage) -> None:
        if image.isNull():
            return
        self._last_frame = image.copy()
        self.frameReady.emit(self._last_frame.copy())

    def _on_backend_error(self, message: str) -> None:
        self.errorOccurred.emit(message)


class QtVideoCaptureBackend(CaptureBackend):
    backend_key = "qt_multimedia"

    def __init__(self) -> None:
        self._camera: QCamera | None = None
        self._session: QMediaCaptureSession | None = None
        self._sink: QVideoSink | None = None

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
        self._sink.videoFrameChanged.connect(lambda frame: _emit_qt_video_frame(frame, frame_callback))
        try:  # pragma: no branch - signal signature depends on Qt runtime
            self._camera.errorOccurred.connect(lambda _error, text: error_callback(text or "USB 相机预览失败"))
        except Exception:
            pass
        self._camera.start()

    def stop_preview(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.deleteLater()
        if self._sink is not None:
            self._sink.deleteLater()
        self._camera = None
        self._session = None
        self._sink = None


class MicroviewCaptureBackend(CaptureBackend):
    backend_key = "microview"
    _PREVIEW_INTERVAL_MS = 33
    _MV_RUN = 1
    _MV_STOP = 0
    _PARAM_BUFFERTYPE = 4
    _PARAM_DISP_PRESENCE = 6
    _PARAM_DISP_WHND = 7
    _PARAM_DISP_TOP = 8
    _PARAM_DISP_LEFT = 9
    _PARAM_GRAB_BITDESCRIBE = 66
    _BUFFER_SYSTEM_MEMORY_GDI = 1
    _SHOW_CLOSE = 0
    _FORMAT_MONO8 = 0
    _FORMAT_RGB1555 = 1
    _FORMAT_RGB24 = 2
    _FORMAT_ARGB8888 = 3
    _FORMAT_RGB565 = 5
    _FORMAT_RGB5515 = 6

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
        self._timer: QTimer | None = None
        self._device_handle: int | None = None
        self._frame_callback: Callable[[QImage], None] | None = None
        self._error_callback: Callable[[str], None] | None = None

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
        frame_callback: Callable[[QImage], None],
        error_callback: Callable[[str], None],
    ) -> None:
        try:
            dll = self._ensure_library()
        except OSError as exc:
            raise RuntimeError(f"Microview SDK 加载失败: {exc}") from exc
        if dll is None:
            raise RuntimeError("未找到 Microview SDK DLL，无法启动实时预览。")
        self.stop_preview()
        self._frame_callback = frame_callback
        self._error_callback = error_callback
        raw_device_handle = dll.MV_OpenDevice(int(device.native_id), True)
        if not raw_device_handle:
            raise RuntimeError(_microview_open_error_message(dll, self._dll_root))
        self._device_handle = int(raw_device_handle)
        self._configure_preview_session(dll, raw_device_handle)
        dll.MV_OperateDevice(self._device_handle, self._MV_RUN)
        self._timer = QTimer()
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._capture_single_frame)
        self._timer.start()

    def stop_preview(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
        self._timer = None
        if self._dll is not None and self._device_handle:
            try:
                self._dll.MV_OperateDevice(self._device_handle, self._MV_STOP)
            except Exception:
                pass
            try:
                self._dll.MV_CloseDevice(self._device_handle)
            except Exception:
                pass
        self._device_handle = None
        self._frame_callback = None
        self._error_callback = None

    def _capture_single_frame(self) -> None:
        if self._dll is None or not self._device_handle or self._frame_callback is None:
            return
        try:
            info = self._MVImageInfo()
            buffer_ptr = self._dll.MV_CaptureSingle(
                self._device_handle,
                True,
                None,
                0,
                ctypes.byref(info),
            )
            if not buffer_ptr:
                return
            image = _microview_buffer_to_qimage(buffer_ptr, info)
        except Exception as exc:
            callback = self._error_callback
            self.stop_preview()
            if callback is not None:
                callback(f"Microview 帧解析失败: {exc}")
            return
        if not image.isNull():
            self._frame_callback(image)

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
        dll.MV_SetDeviceParameter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong]
        dll.MV_SetDeviceParameter.restype = ctypes.c_bool
        dll.MV_CaptureSingle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(self._MVImageInfo),
        ]
        dll.MV_CaptureSingle.restype = ctypes.c_void_p
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

    def _configure_preview_session(self, dll, device_handle) -> None:
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_PRESENCE, self._SHOW_CLOSE)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_WHND, 0)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_LEFT, 0)
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_DISP_TOP, 0)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_BUFFERTYPE, self._BUFFER_SYSTEM_MEMORY_GDI)
        except Exception:
            pass
        try:
            dll.MV_SetDeviceParameter(device_handle, self._PARAM_GRAB_BITDESCRIBE, self._FORMAT_RGB24)
        except Exception:
            pass


def _qt_camera_token(device: QCameraDevice) -> str:
    try:
        raw_id = bytes(device.id())
        if raw_id:
            return raw_id.hex()
    except Exception:
        pass
    description = device.description() or "camera"
    return description.strip().replace(" ", "_")


def _emit_qt_video_frame(frame, frame_callback: Callable[[QImage], None]) -> None:
    if frame is None or not getattr(frame, "isValid", lambda: False)():
        return
    image = frame.toImage()
    if image.isNull():
        return
    frame_callback(image.copy())


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
        return getattr(QImage.Format, "Format_RGBA8888", QImage.Format.Format_ARGB32)
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
    image = QImage(compact_data, width, height, active_stride, image_format)
    if image.isNull():
        raise ValueError(
            "Qt 无法构建 Microview 图像: "
            f"{width}x{height}, stride={active_stride}, format={_microview_pixel_format_name(raw_pixel_format)}"
        )
    return image.copy()
