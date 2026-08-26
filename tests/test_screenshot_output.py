from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QImage

from fdm.screenshot_settings import (
    AfterCaptureTask,
    CollisionPolicy,
    ImageFormat,
    ScreenshotSettings,
)
from fdm.services.screenshot_capture import CaptureMode
from fdm.services.screenshot_output import (
    ScreenshotOutputError,
    ScreenshotOutputService,
    encode_qimage,
)


def _sample_image() -> QImage:
    image = QImage(8, 6, QImage.Format.Format_ARGB32)
    image.fill(QColor(17, 34, 51, 255))
    return image


def test_save_png_uses_template_and_atomic_output(tmp_path: Path) -> None:
    settings = ScreenshotSettings(
        output_directory=str(tmp_path),
        filename_template="Capture_{datetime}_{mode}",
        image_format=ImageFormat.PNG,
    )
    service = ScreenshotOutputService()

    path = service.save_image(
        _sample_image(),
        settings,
        mode=CaptureMode.CU5,
        captured_at=datetime(2026, 8, 10, 12, 34, 56, tzinfo=timezone.utc),
    )

    assert path == tmp_path / "Capture_2026-08-10_12-34-56_cu5.png"
    decoded = QImage(str(path))
    assert (decoded.width(), decoded.height()) == (8, 6)
    assert list(tmp_path.iterdir()) == [path]


def test_collision_policies_increment_fail_and_overwrite(tmp_path: Path) -> None:
    base = ScreenshotSettings(
        output_directory=str(tmp_path),
        filename_template="same",
        collision_policy=CollisionPolicy.INCREMENT,
    )
    service = ScreenshotOutputService()
    first = service.save_image(_sample_image(), base)
    second = service.save_image(_sample_image(), base)

    assert first.name == "same.png"
    assert second.name == "same_001.png"
    with pytest.raises(FileExistsError):
        service.save_image(
            _sample_image(),
            replace(base, collision_policy=CollisionPolicy.FAIL),
        )
    overwritten = service.save_image(
        _sample_image(),
        replace(base, collision_policy=CollisionPolicy.OVERWRITE),
    )
    assert overwritten == first


def test_process_runs_save_and_injected_clipboard_tasks(tmp_path: Path) -> None:
    copied: list[QImage] = []
    service = ScreenshotOutputService(clipboard_setter=copied.append)
    settings = ScreenshotSettings(
        output_directory=str(tmp_path),
        after_capture_tasks=(AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD),
        show_editor=True,
        notification=False,
    )

    result = service.process_capture(_sample_image(), settings)

    assert result.saved_path is not None and result.saved_path.exists()
    assert result.copied_to_clipboard is True
    assert result.open_editor_requested is True
    assert result.notification_requested is False
    assert len(copied) == 1 and not copied[0].isNull()


def test_copy_only_does_not_create_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    copied: list[QImage] = []
    settings = ScreenshotSettings(
        output_directory=str(output),
        after_capture_tasks=(AfterCaptureTask.COPY_CLIPBOARD,),
    )

    result = ScreenshotOutputService(clipboard_setter=copied.append).process_capture(
        _sample_image(), settings
    )

    assert result.saved_path is None
    assert not output.exists()
    assert len(copied) == 1


def test_save_as_uses_explicit_path_suffix_and_configured_encoder_quality(tmp_path: Path) -> None:
    service = ScreenshotOutputService()
    settings = ScreenshotSettings(jpeg_quality=73, png_compression=8)

    jpeg = service.save_image_as(_sample_image(), tmp_path / "chosen.jpg", settings)
    default_png = service.save_image_as(_sample_image(), tmp_path / "without-extension", settings)

    assert jpeg == tmp_path / "chosen.jpg" and jpeg.read_bytes().startswith(b"\xff\xd8")
    assert default_png == tmp_path / "without-extension.png"
    assert default_png.read_bytes().startswith(b"\x89PNG")


def test_save_failure_still_attempts_clipboard_and_returns_partial_error(
    tmp_path: Path,
) -> None:
    blocked_output = tmp_path / "not-a-directory"
    blocked_output.write_text("occupied", encoding="utf-8")
    copied: list[QImage] = []
    settings = ScreenshotSettings(
        output_directory=str(blocked_output),
        after_capture_tasks=(AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD),
    )

    result = ScreenshotOutputService(clipboard_setter=copied.append).process_capture(
        _sample_image(),
        settings,
    )

    assert result.saved_path is None
    assert result.copied_to_clipboard
    assert len(copied) == 1
    assert result.errors and "保存文件失败" in result.failure_summary


def test_clipboard_failure_keeps_successful_file_and_returns_partial_error(
    tmp_path: Path,
) -> None:
    def fail_clipboard(_image: QImage) -> None:
        raise RuntimeError("clipboard busy")

    settings = ScreenshotSettings(
        output_directory=str(tmp_path),
        after_capture_tasks=(AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD),
    )

    result = ScreenshotOutputService(clipboard_setter=fail_clipboard).process_capture(
        _sample_image(),
        settings,
    )

    assert result.saved_path is not None and result.saved_path.exists()
    assert not result.copied_to_clipboard
    assert "clipboard busy" in result.failure_summary


def test_all_output_tasks_failing_raises_aggregated_error(tmp_path: Path) -> None:
    blocked_output = tmp_path / "not-a-directory"
    blocked_output.write_text("occupied", encoding="utf-8")

    def fail_clipboard(_image: QImage) -> None:
        raise RuntimeError("clipboard busy")

    settings = ScreenshotSettings(
        output_directory=str(blocked_output),
        after_capture_tasks=(AfterCaptureTask.SAVE, AfterCaptureTask.COPY_CLIPBOARD),
    )

    with pytest.raises(ScreenshotOutputError) as caught:
        ScreenshotOutputService(clipboard_setter=fail_clipboard).process_capture(
            _sample_image(),
            settings,
        )

    message = str(caught.value)
    assert "截图处理全部失败" in message
    assert "保存文件失败" in message
    assert "clipboard busy" in message


def test_encoders_and_null_image_validation() -> None:
    image = _sample_image()
    for image_format, magic in (
        (ImageFormat.PNG, b"\x89PNG"),
        (ImageFormat.JPEG, b"\xff\xd8"),
        (ImageFormat.WEBP, b"RIFF"),
    ):
        assert encode_qimage(image, image_format=image_format).startswith(magic)

    with pytest.raises(ScreenshotOutputError, match="null"):
        ScreenshotOutputService().process_capture(QImage(), ScreenshotSettings())


def test_png_compression_levels_change_encoding_without_changing_pixels() -> None:
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    for y in range(image.height()):
        for x in range(image.width()):
            red = ((x // 12) * 37 + (y // 16) * 11) & 0xFF
            green = ((x // 20) * 17 + y) & 0xFF
            blue = ((y // 10) * 29 + x) & 0xFF
            image.setPixelColor(x, y, QColor(red, green, blue))

    level_zero = encode_qimage(
        image,
        image_format=ImageFormat.PNG,
        png_compression=0,
    )
    level_nine = encode_qimage(
        image,
        image_format=ImageFormat.PNG,
        png_compression=9,
    )

    assert level_zero != level_nine
    assert len(level_nine) < len(level_zero)
    for payload in (level_zero, level_nine):
        decoded = QImage.fromData(payload, "PNG")
        assert not decoded.isNull()
        assert decoded.convertToFormat(QImage.Format.Format_RGB32) == image
