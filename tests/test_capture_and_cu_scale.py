from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import ctypes
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.services.cu_scale_io import (
    CU_SCALE_MIN_FILE_SIZE,
    CU_SCALE_OFFSET,
    cu_scale_display_name,
    parse_cu_scale_file,
)

try:
    from fdm.services.capture import (
        CaptureBackend,
        CaptureDevice,
        CaptureSessionManager,
        MicroviewCaptureBackend,
        _microview_buffer_to_qimage,
    )
except ModuleNotFoundError:
    CaptureBackend = None
    CaptureDevice = None
    CaptureSessionManager = None
    MicroviewCaptureBackend = None
    _microview_buffer_to_qimage = None


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

    @unittest.skipIf(_microview_buffer_to_qimage is None or MicroviewCaptureBackend is None, "PySide6 not installed")
    def test_microview_rgb24_buffer_converts_to_qimage(self) -> None:
        info = MicroviewCaptureBackend._MVImageInfo()
        info.Width = 2
        info.Heigth = 2
        info.nColor = MicroviewCaptureBackend._FORMAT_RGB24
        info.SkipPixel = 0
        payload = bytes(
            [
                255, 0, 0,
                0, 255, 0,
                0, 0, 255,
                255, 255, 255,
            ]
        )
        info.Length = len(payload)
        buffer = ctypes.create_string_buffer(payload)

        image = _microview_buffer_to_qimage(ctypes.addressof(buffer), info)

        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (2, 2))

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


if __name__ == "__main__":
    unittest.main()
