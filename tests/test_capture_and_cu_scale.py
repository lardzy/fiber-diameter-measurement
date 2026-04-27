from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import ctypes
import json
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.services.cu_scale_io import (
    CU_SCALE_MIN_FILE_SIZE,
    CU_SCALE_OFFSET,
    cu_scale_display_name,
    format_cu_scale_record_summary,
    parse_cu_scale_file,
)

try:
    import fdm.services.capture as capture_module
    from fdm.services.capture import (
        CaptureBackend,
        CaptureDevice,
        CaptureSessionManager,
        MicroviewCaptureBackend,
        _microview_buffer_to_qimage,
    )
except ModuleNotFoundError:
    capture_module = None
    CaptureBackend = None
    CaptureDevice = None
    CaptureSessionManager = None
    MicroviewCaptureBackend = None
    _microview_buffer_to_qimage = None

try:
    from PySide6.QtGui import QColor, QImage
except ModuleNotFoundError:
    QColor = None
    QImage = None


class CaptureAndCuScaleTests(unittest.TestCase):
    def test_cu_scale_display_name_strips_date_suffix(self) -> None:
        self.assertEqual(cu_scale_display_name("40x-2025.09.15.scl"), "40x")
        self.assertEqual(cu_scale_display_name("40x.scl"), "40x")

    def test_parse_cu_scale_file_builds_um_preset(self) -> None:
        payload = bytearray(CU_SCALE_MIN_FILE_SIZE)
        struct.pack_into("<f", payload, CU_SCALE_OFFSET, 0.25)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "20x-2025.09.15.scl"
            path.write_bytes(bytes(payload))
            record = parse_cu_scale_file(path)

        self.assertEqual(record.preset.name, "20x")
        self.assertEqual(record.preset.unit, "um")
        self.assertAlmostEqual(record.preset.resolved_pixels_per_unit(), 4.0)
        self.assertIsNone(record.preset.pixel_distance)
        self.assertIsNone(record.preset.actual_distance)

    def test_parse_cu_scale_file_accepts_legacy_trailing_bytes(self) -> None:
        payload = bytearray(CU_SCALE_MIN_FILE_SIZE + 8)
        struct.pack_into("<f", payload, CU_SCALE_OFFSET, 0.5)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "10x-legacy.scl"
            path.write_bytes(bytes(payload))
            record = parse_cu_scale_file(path)

        self.assertEqual(record.preset.name, "10x-legacy")
        self.assertAlmostEqual(record.preset.resolved_pixels_per_unit(), 2.0)

    def test_format_cu_scale_record_summary_contains_scale_values(self) -> None:
        payload = bytearray(CU_SCALE_MIN_FILE_SIZE)
        struct.pack_into("<f", payload, CU_SCALE_OFFSET, 0.25)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "20x-2025.09.15.scl"
            path.write_bytes(bytes(payload))
            record = parse_cu_scale_file(path)

        summary = format_cu_scale_record_summary(record)

        self.assertIn("文件: 20x-2025.09.15.scl", summary)
        self.assertIn("预设名称: 20x", summary)
        self.assertIn("标尺: 4 px/um", summary)
        self.assertIn("换算: 0.25 um/px", summary)

    def test_parse_cu_scale_file_rejects_invalid_payload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.scl"
            path.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                parse_cu_scale_file(path)

    @unittest.skipIf(CaptureSessionManager is None, "PySide6 not installed")
    def test_capture_session_manager_ignores_backend_enumeration_failure(self) -> None:
        class BrokenBackend(CaptureBackend):
            backend_key = "microview"

            def list_devices(self) -> list[CaptureDevice]:
                raise RuntimeError("sdk load failed")

            def start_preview(self, device, *, frame_callback, error_callback) -> None:
                raise NotImplementedError

            def stop_preview(self) -> None:
                return None

        class WorkingBackend(CaptureBackend):
            backend_key = "qt_multimedia"

            def list_devices(self) -> list[CaptureDevice]:
                return [
                    CaptureDevice(
                        id="qt_multimedia:usb-1",
                        name="USB Camera",
                        backend_key=self.backend_key,
                        native_id="usb-1",
                    )
                ]

            def start_preview(self, device, *, frame_callback, error_callback) -> None:
                raise NotImplementedError

            def stop_preview(self) -> None:
                return None

        manager = CaptureSessionManager(
            backends=[BrokenBackend(), WorkingBackend()],
            refresh_on_init=False,
        )

        devices = manager.refresh_devices()

        self.assertEqual([device.id for device in devices], ["qt_multimedia:usb-1"])
        self.assertEqual(manager.selected_device_id(), "qt_multimedia:usb-1")
        self.assertEqual(manager.device_refresh_warnings(), ["Microview SDK: sdk load failed"])

    @unittest.skipIf(CaptureSessionManager is None or QImage is None or QColor is None, "PySide6 not installed")
    def test_capture_session_manager_ignores_stale_frames_after_stop(self) -> None:
        class WorkingBackend(CaptureBackend):
            backend_key = "qt_multimedia"

            def list_devices(self) -> list[CaptureDevice]:
                return [
                    CaptureDevice(
                        id="qt_multimedia:usb-1",
                        name="USB Camera",
                        backend_key=self.backend_key,
                        native_id="usb-1",
                    )
                ]

            def start_preview(self, device, *, preview_target=None, frame_callback, error_callback) -> None:
                return None

            def stop_preview(self) -> None:
                return None

        manager = CaptureSessionManager(backends=[WorkingBackend()], refresh_on_init=False)
        manager.refresh_devices()
        received: list[tuple[int, int]] = []
        manager.frameReady.connect(lambda image: received.append((image.width(), image.height())))

        self.assertTrue(manager.start_preview())
        generation = manager._preview_generation
        first_frame = QImage(32, 24, QImage.Format.Format_RGB32)
        first_frame.fill(QColor("#88CCEE"))
        manager._on_frame_ready(generation, first_frame)

        self.assertEqual(received, [(32, 24)])
        self.assertIsNotNone(manager.last_frame())

        manager.stop_preview()
        stale_frame = QImage(48, 36, QImage.Format.Format_RGB32)
        stale_frame.fill(QColor("#F4D35E"))
        manager._on_frame_ready(generation, stale_frame)

        self.assertEqual(received, [(32, 24)])
        self.assertIsNone(manager.last_frame())

    @unittest.skipIf(CaptureSessionManager is None or QImage is None or QColor is None, "PySide6 not installed")
    def test_capture_session_manager_routes_analysis_frames(self) -> None:
        class WorkingBackend(CaptureBackend):
            backend_key = "microview"

            def __init__(self) -> None:
                self.request_ids: list[int] = []

            def list_devices(self) -> list[CaptureDevice]:
                return [
                    CaptureDevice(
                        id="microview:0",
                        name="Microview #1",
                        backend_key=self.backend_key,
                        native_id=0,
                    )
                ]

            def preview_kind(self, device: CaptureDevice) -> str:
                return "native_embed"

            def start_preview(self, device, *, preview_target=None, frame_callback, error_callback) -> None:
                return None

            def stop_preview(self) -> None:
                return None

            def can_request_analysis_frame(self, device: CaptureDevice) -> bool:
                return True

            def request_analysis_frame(self, device, *, request_id: int, frame_callback, error_callback) -> None:
                self.request_ids.append(request_id)
                image = QImage(64, 48, QImage.Format.Format_RGB32)
                image.fill(QColor("#88CCEE"))
                frame_callback(request_id, image)

        backend = WorkingBackend()
        manager = CaptureSessionManager(backends=[backend], refresh_on_init=False)
        manager.refresh_devices()
        self.assertTrue(manager.start_preview(preview_target=object()))
        received: list[tuple[int, int, int]] = []
        manager.analysisFrameReady.connect(
            lambda request_id, image: received.append((request_id, image.width(), image.height()))
        )

        self.assertTrue(manager.request_analysis_frame(7))

        self.assertEqual(backend.request_ids, [7])
        self.assertEqual(received, [(7, 64, 48)])

    @unittest.skipIf(capture_module is None, "PySide6 not installed")
    def test_microview_isolated_backend_update_preview_target_sends_helper_command(self) -> None:
        class FakePreviewTarget:
            def native_preview_handle(self) -> int:
                return 12345

            def native_preview_size(self) -> tuple[int, int]:
                return 768, 576

        class FakeProcess:
            def __init__(self) -> None:
                self.payloads: list[bytes] = []

            def state(self):
                return capture_module.QProcess.ProcessState.Running

            def write(self, payload: bytes) -> int:
                self.payloads.append(payload)
                return len(payload)

            def waitForBytesWritten(self, timeout: int) -> bool:
                del timeout
                return True

        backend = capture_module.MicroviewIsolatedBackend()
        fake_process = FakeProcess()
        backend._preview_process = fake_process
        backend._preview_started = True

        backend.update_preview_target(FakePreviewTarget())

        self.assertEqual(len(fake_process.payloads), 1)
        payload = json.loads(fake_process.payloads[0].decode("utf-8"))
        self.assertEqual(
            payload,
            {
                "command": "update_target",
                "hwnd": 12345,
                "width": 768,
                "height": 576,
            },
        )

    @unittest.skipIf(capture_module is None, "PySide6 not installed")
    def test_microview_helper_command_forwards_gdi_preview_preference(self) -> None:
        class FakePreviewTarget:
            def native_preview_handle(self) -> int:
                return 12345

            def native_preview_size(self) -> tuple[int, int]:
                return 768, 576

            def prefer_gdi_preview(self) -> bool:
                return True

        _, arguments = capture_module._microview_helper_command(  # noqa: SLF001
            "preview",
            device_index=0,
            preview_target=FakePreviewTarget(),
        )

        self.assertIn("--prefer-gdi", arguments)

    @unittest.skipIf(capture_module is None, "PySide6 not installed")
    def test_microview_isolated_backend_snapshot_command_is_sent(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.payloads: list[bytes] = []

            def state(self):
                return capture_module.QProcess.ProcessState.Running

            def write(self, payload: bytes) -> int:
                self.payloads.append(payload)
                return len(payload)

            def waitForBytesWritten(self, timeout: int) -> bool:
                del timeout
                return True

        backend = capture_module.MicroviewIsolatedBackend()
        backend._preview_process = FakeProcess()
        backend._preview_started = True
        requested: list[int] = []

        backend.request_analysis_frame(
            capture_module.CaptureDevice(id="microview:0", name="Microview #1", backend_key="microview", native_id=0),
            request_id=11,
            frame_callback=lambda request_id, image: requested.append(request_id),
            error_callback=lambda request_id, message: requested.append(-request_id),
        )

        self.assertEqual(requested, [])
        self.assertEqual(len(backend._preview_process.payloads), 1)
        payload = json.loads(backend._preview_process.payloads[0].decode("utf-8"))
        self.assertEqual(payload, {"command": "snapshot", "request_id": 11})

    @unittest.skipIf(_microview_buffer_to_qimage is None or MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_rgb24_buffer_converts_to_qimage(self) -> None:
        info = MicroviewCaptureBackend._MVImageInfo()
        info.Width = 2
        info.Heigth = 2
        info.nColor = MicroviewCaptureBackend._FORMAT_RGB24
        info.SkipPixel = 0
        payload = bytes(
            [
                0, 0, 255,
                0, 255, 0,
                255, 0, 0,
                255, 255, 255,
            ]
        )
        info.Length = len(payload)
        buffer = ctypes.create_string_buffer(payload)

        image = _microview_buffer_to_qimage(ctypes.addressof(buffer), info)

        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (2, 2))
        self.assertEqual(image.pixelColor(0, 0).red(), 255)
        self.assertEqual(image.pixelColor(0, 0).blue(), 0)

    @unittest.skipIf(_microview_buffer_to_qimage is None or MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_24bpp_alias_converts_to_qimage(self) -> None:
        info = MicroviewCaptureBackend._MVImageInfo()
        info.Width = 2
        info.Heigth = 2
        info.nColor = 24
        info.SkipPixel = 0
        payload = bytes(
            [
                0, 0, 255,
                0, 255, 0,
                255, 0, 0,
                255, 255, 255,
            ]
        )
        info.Length = len(payload)
        buffer = ctypes.create_string_buffer(payload)

        image = _microview_buffer_to_qimage(ctypes.addressof(buffer), info)

        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (2, 2))

    @unittest.skipIf(_microview_buffer_to_qimage is None or MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_32bpp_argb_buffer_converts_to_qimage(self) -> None:
        info = MicroviewCaptureBackend._MVImageInfo()
        info.Width = 2
        info.Heigth = 1
        info.nColor = 32
        info.SkipPixel = 0
        payload = bytes(
            [
                0, 0, 255, 255,
                0, 255, 0, 255,
            ]
        )
        info.Length = len(payload)
        buffer = ctypes.create_string_buffer(payload)

        image = _microview_buffer_to_qimage(ctypes.addressof(buffer), info)

        self.assertFalse(image.isNull())
        self.assertEqual(image.pixelColor(0, 0).red(), 255)
        self.assertEqual(image.pixelColor(0, 0).blue(), 0)
        self.assertEqual(image.pixelColor(0, 0).alpha(), 255)

    @unittest.skipIf(_microview_buffer_to_qimage is None or MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_32bpp_zero_alpha_buffer_is_forced_opaque(self) -> None:
        info = MicroviewCaptureBackend._MVImageInfo()
        info.Width = 1
        info.Heigth = 1
        info.nColor = 32
        info.SkipPixel = 0
        payload = bytes([0, 0, 255, 0])
        info.Length = len(payload)
        buffer = ctypes.create_string_buffer(payload)

        image = _microview_buffer_to_qimage(ctypes.addressof(buffer), info)

        self.assertFalse(image.isNull())
        self.assertEqual(image.pixelColor(0, 0).red(), 255)
        self.assertEqual(image.pixelColor(0, 0).alpha(), 255)

    @unittest.skipIf(_microview_buffer_to_qimage is None or MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_buffer_rejects_short_payload_before_qimage_construction(self) -> None:
        info = MicroviewCaptureBackend._MVImageInfo()
        info.Width = 3
        info.Heigth = 2
        info.nColor = MicroviewCaptureBackend._FORMAT_RGB24
        info.SkipPixel = 0
        payload = bytes([0] * 8)
        info.Length = len(payload)
        buffer = ctypes.create_string_buffer(payload)

        with self.assertRaises(ValueError):
            _microview_buffer_to_qimage(ctypes.addressof(buffer), info)

    @unittest.skipIf(MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_frame_mode_uses_work_skip_for_levin_board(self) -> None:
        backend = MicroviewCaptureBackend()
        calls: list[tuple[int, int]] = []

        class FakeDll:
            def MV_SetDeviceParameter(self, handle, parameter, value):
                calls.append((int(parameter), int(value)))
                return True

        backend._apply_frame_mode(FakeDll(), 1, MicroviewCaptureBackend._LEVIN_M20)

        self.assertEqual(
            calls,
            [
                (
                    MicroviewCaptureBackend._PARAM_WORK_SKIP,
                    MicroviewCaptureBackend._INTERLUDE,
                )
            ],
        )

    @unittest.skipIf(MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_frame_mode_uses_work_field_for_non_levin_board(self) -> None:
        backend = MicroviewCaptureBackend()
        calls: list[tuple[int, int]] = []

        class FakeDll:
            def MV_SetDeviceParameter(self, handle, parameter, value):
                calls.append((int(parameter), int(value)))
                return True

        backend._apply_frame_mode(FakeDll(), 1, 0x12345678)

        self.assertEqual(
            calls,
            [
                (
                    MicroviewCaptureBackend._PARAM_WORK_FIELD,
                    MicroviewCaptureBackend._COLLECTION_FRAME,
                )
            ],
        )

    @unittest.skipIf(MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_stop_preview_releases_display_and_capture_state(self) -> None:
        backend = MicroviewCaptureBackend()
        calls: list[tuple[str, int, int]] = []

        class FakeDll:
            def MV_SetDeviceParameter(self, handle, parameter, value):
                calls.append(("set", int(parameter), int(value)))
                return True

            def MV_OperateDevice(self, handle, operation):
                calls.append(("operate", int(operation), 0))
                return 0

            def MV_CloseDevice(self, handle):
                calls.append(("close", int(handle), 0))
                return None

        backend._dll = FakeDll()
        backend._device_handle = 9
        backend._device_index = 0
        backend._preview_resolution = (1760, 1328)

        backend.stop_preview()

        self.assertEqual(
            calls,
            [
                ("set", MicroviewCaptureBackend._PARAM_DISP_PRESENCE, MicroviewCaptureBackend._SHOW_CLOSE),
                ("set", MicroviewCaptureBackend._PARAM_DISP_WHND, 0),
                ("set", MicroviewCaptureBackend._PARAM_DISP_LEFT, 0),
                ("set", MicroviewCaptureBackend._PARAM_DISP_TOP, 0),
                ("set", MicroviewCaptureBackend._PARAM_RESTOPCAPTURE, 0),
                ("operate", MicroviewCaptureBackend._MV_STOP, 0),
                ("close", 9, 0),
            ],
        )
        self.assertIsNone(backend._device_handle)

    @unittest.skipIf(CaptureSessionManager is None, "PySide6 not installed")
    def test_capture_session_manager_recreates_microview_backend_after_stop(self) -> None:
        class NativeBackend(CaptureBackend):
            backend_key = "microview"

            def list_devices(self) -> list[CaptureDevice]:
                return [
                    CaptureDevice(
                        id="microview:0",
                        name="Microview #1",
                        backend_key=self.backend_key,
                        native_id=0,
                    )
                ]

            def preview_kind(self, device: CaptureDevice) -> str:
                return "native_embed"

            def start_preview(self, device, *, preview_target=None, frame_callback, error_callback) -> None:
                return None

            def stop_preview(self) -> None:
                return None

        backend = NativeBackend()
        manager = CaptureSessionManager(backends=[backend], refresh_on_init=False)
        manager.refresh_devices()

        self.assertTrue(manager.start_preview(preview_target=object()))
        manager.stop_preview()

        self.assertIsNot(manager._backends[0], backend)
        self.assertIsInstance(manager._backends[0], NativeBackend)

    @unittest.skipIf(CaptureSessionManager is None or QImage is None or QColor is None, "PySide6 not installed")
    def test_capture_session_manager_does_not_cache_native_embed_still_frame(self) -> None:
        still_frame = QImage(64, 48, QImage.Format.Format_RGB32)
        still_frame.fill(QColor("#88CCEE"))

        class NativeBackend(CaptureBackend):
            backend_key = "microview"

            def __init__(self) -> None:
                self.capture_calls = 0

            def list_devices(self) -> list[CaptureDevice]:
                return [
                    CaptureDevice(
                        id="microview:0",
                        name="Microview #1",
                        backend_key=self.backend_key,
                        native_id=0,
                    )
                ]

            def preview_kind(self, device: CaptureDevice) -> str:
                return "native_embed"

            def start_preview(self, device, *, preview_target=None, frame_callback, error_callback) -> None:
                return None

            def stop_preview(self) -> None:
                return None

            def can_capture_still(self, device: CaptureDevice) -> bool:
                return True

            def capture_still_frame(self, device: CaptureDevice) -> QImage | None:
                self.capture_calls += 1
                return still_frame.copy()

        backend = NativeBackend()
        manager = CaptureSessionManager(backends=[backend], refresh_on_init=False)
        manager.refresh_devices()

        captured = manager.capture_still_frame()

        self.assertIsNotNone(captured)
        self.assertEqual((captured.width(), captured.height()), (64, 48))
        self.assertEqual(backend.capture_calls, 1)
        self.assertIsNone(manager.last_frame())

    @unittest.skipIf(
        capture_module is None or _microview_buffer_to_qimage is None or MicroviewCaptureBackend is None,
        "PySide6 not installed",
    )
    def test_microview_capture_single_uses_preallocated_buffer_when_sdk_returns_null_pointer(self) -> None:
        backend = MicroviewCaptureBackend()
        payload = bytes([0, 0, 255, 0, 255, 0])

        class FakeDll:
            def MV_SetDeviceParameter(self, handle, parameter, value):
                if int(parameter) == MicroviewCaptureBackend._PARAM_SET_GARBIMAGEINFO:
                    info = ctypes.cast(int(value), ctypes.POINTER(MicroviewCaptureBackend._MVImageInfo)).contents
                    info.Length = len(payload)
                    info.Width = 2
                    info.Heigth = 1
                    info.nColor = 24
                    info.SkipPixel = 0
                return True

            def MV_CaptureSingle(self, handle, process, buffer_ptr, buffer_len, info_ptr):
                ctypes.memmove(int(buffer_ptr), payload, len(payload))
                return 0

        backend._dll = FakeDll()

        image = backend._capture_single_frame_image(handle=1, process=False)

        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (2, 1))
        self.assertEqual(image.pixelColor(0, 0).red(), 255)

    @unittest.skipIf(MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_failed_capture_records_diagnostics(self) -> None:
        backend = MicroviewCaptureBackend()

        class FakeDll:
            def MV_SetDeviceParameter(self, handle, parameter, value):
                if int(parameter) == MicroviewCaptureBackend._PARAM_SET_GARBIMAGEINFO:
                    info = ctypes.cast(int(value), ctypes.POINTER(MicroviewCaptureBackend._MVImageInfo)).contents
                    info.Length = 0
                    info.Width = 0
                    info.Heigth = 0
                    info.nColor = 24
                    info.SkipPixel = 0
                return True

            def MV_CaptureSingle(self, handle, process, buffer_ptr, buffer_len, info_ptr):
                return 0

            def MV_GetLastError(self, display):
                return 12

        backend._dll = FakeDll()

        image = backend._capture_single_frame_image(handle=1, process=False)

        self.assertTrue(image.isNull())
        diagnostics = backend.last_capture_diagnostics()
        self.assertIn("attempt=1", diagnostics)
        self.assertIn("sdk_error_after=12", diagnostics)


if __name__ == "__main__":
    unittest.main()
