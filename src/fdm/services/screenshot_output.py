from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

from PySide6.QtCore import QByteArray, QBuffer, QIODeviceBase
from PySide6.QtGui import QGuiApplication, QImage, QImageWriter

from fdm.atomic_io import atomic_write_bytes
from fdm.screenshot_settings import (
    AfterCaptureTask,
    CollisionPolicy,
    ImageFormat,
    ScreenshotSettings,
)
from fdm.services.screenshot_capture import CaptureMode


ClipboardSetter = Callable[[QImage], None]


class ScreenshotOutputError(RuntimeError):
    pass


class ScreenshotEncodingError(ScreenshotOutputError):
    pass


class ScreenshotClipboardError(ScreenshotOutputError):
    pass


@dataclass(frozen=True, slots=True)
class OutputResult:
    saved_path: Path | None = None
    copied_to_clipboard: bool = False
    open_editor_requested: bool = False
    notification_requested: bool = False
    errors: tuple[str, ...] = ()

    @property
    def failure_summary(self) -> str:
        return "；".join(self.errors)


class ScreenshotOutputService:
    """Encode and dispatch one capture according to normalized user settings."""

    def __init__(self, *, clipboard_setter: ClipboardSetter | None = None) -> None:
        self._clipboard_setter = clipboard_setter

    def process_capture(
        self,
        image: QImage,
        settings: ScreenshotSettings,
        *,
        mode: CaptureMode = CaptureMode.REGION,
        captured_at: datetime | None = None,
    ) -> OutputResult:
        if not isinstance(image, QImage) or image.isNull():
            raise ScreenshotOutputError("cannot publish a null screenshot")
        normalized = settings.normalized()
        saved_path: Path | None = None
        copied = False
        errors: list[str] = []
        successes = 0
        for task in normalized.after_capture_tasks:
            if task is AfterCaptureTask.SAVE:
                try:
                    saved_path = self.save_image(
                        image,
                        normalized,
                        mode=mode,
                        captured_at=captured_at,
                    )
                except Exception as exc:  # noqa: BLE001 - independent output task
                    errors.append(
                        f"保存文件失败：{str(exc).strip() or type(exc).__name__}"
                    )
                else:
                    successes += 1
            elif task is AfterCaptureTask.COPY_CLIPBOARD:
                try:
                    self.copy_to_clipboard(image)
                except Exception as exc:  # noqa: BLE001 - independent output task
                    errors.append(
                        f"复制到剪贴板失败：{str(exc).strip() or type(exc).__name__}"
                    )
                else:
                    copied = True
                    successes += 1
        if successes == 0:
            summary = "；".join(errors) or "未执行任何截图输出任务"
            raise ScreenshotOutputError(f"截图处理全部失败：{summary}")
        return OutputResult(
            saved_path=saved_path,
            copied_to_clipboard=copied,
            open_editor_requested=normalized.show_editor,
            notification_requested=normalized.notification,
            errors=tuple(errors),
        )

    # Both names read naturally at integration call sites.
    publish = process_capture
    process = process_capture

    def save_image(
        self,
        image: QImage,
        settings: ScreenshotSettings,
        *,
        mode: CaptureMode = CaptureMode.REGION,
        captured_at: datetime | None = None,
    ) -> Path:
        if not isinstance(image, QImage) or image.isNull():
            raise ScreenshotOutputError("cannot save a null screenshot")
        normalized = settings.normalized()
        output_directory = Path(normalized.output_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        target = self.resolve_output_path(
            normalized,
            mode=mode,
            captured_at=captured_at,
        )
        encoded = encode_qimage(
            image,
            image_format=normalized.image_format,
            png_compression=normalized.png_compression,
            jpeg_quality=normalized.jpeg_quality,
            webp_quality=normalized.webp_quality,
        )
        atomic_write_bytes(target, encoded)
        return target

    def resolve_output_path(
        self,
        settings: ScreenshotSettings,
        *,
        mode: CaptureMode = CaptureMode.REGION,
        captured_at: datetime | None = None,
    ) -> Path:
        normalized = settings.normalized()
        output_directory = Path(normalized.output_directory).expanduser()
        timestamp = captured_at or datetime.now().astimezone()
        if timestamp.tzinfo is None:
            timestamp = timestamp.astimezone()

        def candidate(counter: int) -> Path:
            name = _render_filename(
                normalized.filename_template,
                mode=mode,
                captured_at=timestamp,
                counter=counter,
                image_format=normalized.image_format,
            )
            if counter and "{counter" not in normalized.filename_template:
                name = f"{Path(name).stem}_{counter:03d}{normalized.image_format.suffix}"
            return output_directory / name

        first = candidate(0)
        if normalized.collision_policy is CollisionPolicy.OVERWRITE:
            return first
        if normalized.collision_policy is CollisionPolicy.FAIL:
            if first.exists():
                raise FileExistsError(first)
            return first
        for counter in range(0, 100_000):
            path = candidate(counter)
            if not path.exists():
                return path
        raise ScreenshotOutputError("could not allocate a unique screenshot filename")

    def copy_to_clipboard(self, image: QImage) -> None:
        if not isinstance(image, QImage) or image.isNull():
            raise ScreenshotClipboardError("cannot copy a null screenshot")
        copied_image = image.copy()
        if self._clipboard_setter is not None:
            self._clipboard_setter(copied_image)
            return
        app = QGuiApplication.instance()
        if app is None:
            raise ScreenshotClipboardError(
                "clipboard output requires a running QGuiApplication"
            )
        app.clipboard().setImage(copied_image)


def encode_qimage(
    image: QImage,
    *,
    image_format: ImageFormat = ImageFormat.PNG,
    png_compression: int = 6,
    jpeg_quality: int = 92,
    webp_quality: int = 90,
) -> bytes:
    if not isinstance(image, QImage) or image.isNull():
        raise ScreenshotEncodingError("cannot encode a null screenshot")
    image_format = ImageFormat.parse(image_format, default=ImageFormat.PNG)
    buffer = QBuffer()
    if not buffer.open(QIODeviceBase.OpenModeFlag.WriteOnly):
        raise ScreenshotEncodingError("could not open the screenshot encoder buffer")
    writer = QImageWriter(buffer, QByteArray(image_format.qt_format))
    if image_format is ImageFormat.PNG:
        compression = max(0, min(9, int(png_compression)))
        # QImageWriter's public compression option is a 0..100 ratio even
        # though our persisted PNG preference follows zlib's conventional
        # 0..9 level.
        writer.setCompression(round(compression * 100 / 9))
        # Qt PNG plugins differ in which generic writer option they honor, so
        # provide both forms. Quality is inverse here: 100 means
        # fastest/largest, while 0 maps to the strongest PNG compression.
        writer.setQuality(round((9 - compression) * 100 / 9))
    elif image_format is ImageFormat.JPEG:
        writer.setQuality(max(1, min(100, int(jpeg_quality))))
    elif image_format is ImageFormat.WEBP:
        writer.setQuality(max(1, min(100, int(webp_quality))))
    if not writer.write(image):
        error = writer.errorString() or f"{image_format.value} encoder is unavailable"
        buffer.close()
        raise ScreenshotEncodingError(error)
    payload = bytes(buffer.data())
    buffer.close()
    if not payload:
        raise ScreenshotEncodingError("screenshot encoder returned no data")
    return payload


def _render_filename(
    template: str,
    *,
    mode: CaptureMode,
    captured_at: datetime,
    counter: int,
    image_format: ImageFormat,
) -> str:
    if not isinstance(mode, CaptureMode):
        mode = CaptureMode.parse(mode)
    values = {
        "date": captured_at.strftime("%Y-%m-%d"),
        "time": captured_at.strftime("%H-%M-%S"),
        "datetime": captured_at.strftime("%Y-%m-%d_%H-%M-%S"),
        "mode": mode.value,
        "counter": counter,
    }
    try:
        rendered = template.format_map(values)
    except (KeyError, ValueError, IndexError):
        rendered = f"Screenshot_{values['date']}_{values['time']}"
    rendered = Path(rendered).name
    rendered = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", rendered)
    rendered = re.sub(r"\s+", " ", rendered).strip(" .")
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        if rendered.lower().endswith(suffix):
            rendered = rendered[: -len(suffix)].rstrip(" .")
            break
    if not rendered:
        rendered = "Screenshot"
    if rendered.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        rendered = f"_{rendered}"
    return f"{rendered}{image_format.suffix}"


__all__ = [
    "ClipboardSetter",
    "OutputResult",
    "ScreenshotClipboardError",
    "ScreenshotEncodingError",
    "ScreenshotOutputError",
    "ScreenshotOutputService",
    "encode_qimage",
]
