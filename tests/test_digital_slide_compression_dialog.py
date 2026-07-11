from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtWidgets import QApplication, QGroupBox

    from fdm.settings import AppSettings
    from fdm.ui.dialogs import DigitalSlideCompressionDialog, SettingsDialog

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _FakeCompressionWorker:
    instances: list["_FakeCompressionWorker"] = []

    def __init__(self, source: Path, target: Path, *, codec: str, quality: int | None) -> None:
        self.source = source
        self.target = target
        self.codec = codec
        self.quality = quality
        self.progress = _FakeSignal()
        self.finished = _FakeSignal()
        self.failed = _FakeSignal()
        self.started = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class DigitalSlideCompressionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        _FakeCompressionWorker.instances.clear()

    def test_prefills_source_and_default_copy_path(self) -> None:
        source = Path("/tmp/sample.fdmslide")
        dialog = DigitalSlideCompressionDialog(
            AppSettings(digital_slide_capture_jpeg_quality=87),
            source_path=source,
        )
        try:
            self.assertEqual(dialog.windowTitle(), "压缩数字化切片副本")
            self.assertEqual(dialog.source_path(), source)
            self.assertEqual(dialog.target_path(), Path("/tmp/sample_compressed.fdmslide"))
            self.assertEqual(dialog._codec_combo.currentData(), "jpeg")  # noqa: SLF001
            self.assertEqual(dialog._quality_slider.value(), 87)  # noqa: SLF001
            self.assertFalse(dialog.is_running())
            self.assertIsNone(dialog.completed_path())
        finally:
            dialog.close()

    def test_settings_page_no_longer_contains_compression_task_controls(self) -> None:
        dialog = SettingsDialog(AppSettings(), document=None)
        try:
            digital_slide_page = dialog._settings_pages.widget(5)  # noqa: SLF001
            titles = [group.title() for group in digital_slide_page.findChildren(QGroupBox)]
            self.assertNotIn("切片压缩工具", titles)
            self.assertFalse(hasattr(dialog, "_digital_slide_compress_start_button"))
            self.assertFalse(hasattr(dialog, "_digital_slide_compression_worker"))
        finally:
            dialog.close()

    def test_validation_rejects_missing_invalid_and_same_source(self) -> None:
        dialog = DigitalSlideCompressionDialog(AppSettings())
        try:
            with patch("fdm.ui.dialogs.QMessageBox.information") as information:
                self.assertFalse(dialog.start_compression())
            information.assert_called_once()

            dialog._source_edit.setText("/tmp/not-a-slide.txt")  # noqa: SLF001
            with patch("fdm.ui.dialogs.QMessageBox.warning") as warning:
                self.assertFalse(dialog.start_compression())
            warning.assert_called_once()

            with TemporaryDirectory() as tmp_dir:
                source = Path(tmp_dir) / "source.fdmslide"
                source.write_bytes(b"placeholder")
                dialog.set_source_path(source)
                dialog._target_edit.setText(str(source))  # noqa: SLF001
                with patch("fdm.ui.dialogs.QMessageBox.warning") as warning:
                    self.assertFalse(dialog.start_compression())
                self.assertIn("不能与源文件相同", warning.call_args.args[2])
        finally:
            dialog.close()

    def test_starts_worker_reports_progress_and_exposes_completed_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.fdmslide"
            source.write_bytes(b"placeholder")
            target = Path(tmp_dir) / "archive.fdmslide"
            dialog = DigitalSlideCompressionDialog(AppSettings(), source_path=source)
            dialog._target_edit.setText(str(target))  # noqa: SLF001
            completed: list[str] = []
            dialog.compression_finished.connect(completed.append)
            try:
                with (
                    patch("fdm.ui.dialogs.DigitalSlideCompressionWorker", _FakeCompressionWorker),
                    patch("fdm.ui.dialogs.QMessageBox.information"),
                ):
                    self.assertTrue(dialog.start_compression())
                    worker = _FakeCompressionWorker.instances[-1]
                    self.assertTrue(worker.started)
                    self.assertEqual(worker.source, source)
                    self.assertEqual(worker.target, target)
                    self.assertEqual(worker.codec, "jpeg")
                    self.assertEqual(worker.quality, 90)
                    self.assertTrue(dialog.is_running())
                    self.assertFalse(dialog._source_edit.isEnabled())  # noqa: SLF001

                    worker.progress.emit(3, 10)
                    self.assertEqual(dialog._progress.maximum(), 10)  # noqa: SLF001
                    self.assertEqual(dialog._progress.value(), 3)  # noqa: SLF001
                    worker.finished.emit(str(target))

                self.assertFalse(dialog.is_running())
                self.assertEqual(dialog.completed_path(), target)
                self.assertEqual(completed, [str(target)])
                self.assertTrue(dialog._source_edit.isEnabled())  # noqa: SLF001
                self.assertEqual(dialog._progress.format(), "压缩完成")  # noqa: SLF001
            finally:
                dialog.close()

    def test_png_target_gets_suffix_and_does_not_pass_quality(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.fdmslide"
            source.write_bytes(b"placeholder")
            dialog = DigitalSlideCompressionDialog(AppSettings(), source_path=source)
            dialog._target_edit.setText(str(Path(tmp_dir) / "lossless-copy"))  # noqa: SLF001
            dialog._codec_combo.setCurrentIndex(dialog._codec_combo.findData("png"))  # noqa: SLF001
            try:
                with patch("fdm.ui.dialogs.DigitalSlideCompressionWorker", _FakeCompressionWorker):
                    self.assertTrue(dialog.start_compression())
                worker = _FakeCompressionWorker.instances[-1]
                self.assertEqual(worker.target, Path(tmp_dir) / "lossless-copy.fdmslide")
                self.assertEqual(worker.codec, "png")
                self.assertIsNone(worker.quality)
                with patch("fdm.ui.dialogs.QMessageBox.warning"):
                    worker.failed.emit("test cleanup")
            finally:
                dialog._running = False  # noqa: SLF001
                dialog.close()

    def test_running_task_blocks_dialog_close(self) -> None:
        dialog = DigitalSlideCompressionDialog(AppSettings())
        try:
            dialog.show()
            self.app.processEvents()
            dialog._running = True  # noqa: SLF001
            with patch("fdm.ui.dialogs.QMessageBox.information") as information:
                dialog.reject()
            self.assertTrue(dialog.isVisible())
            information.assert_called_once()
        finally:
            dialog._running = False  # noqa: SLF001
            dialog.close()


if __name__ == "__main__":
    unittest.main()
