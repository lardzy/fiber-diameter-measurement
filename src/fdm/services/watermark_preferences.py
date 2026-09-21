"""Last accepted watermark and its Logo, stored in the writable user profile."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fdm import settings
from fdm.atomic_io import atomic_write_bytes
from fdm.services.watermark_assets import verify_logo
from fdm.watermark import WatermarkSpec


def _logo_path(spec: WatermarkSpec, settings_path: Path) -> Path:
    return settings_path.parent / "watermark-assets" / f"{spec.logo_sha256}.png"


def load_watermark_default_assets(
    spec: WatermarkSpec | None, *, settings_path: Path | None = None,
) -> dict[str, bytes]:
    if spec is None or spec.kind != "logo" or not spec.logo_sha256:
        return {}
    target = settings_path if settings_path is not None else settings.settings_file_path()
    data = _logo_path(spec, target).read_bytes()
    verify_logo(data, spec.logo_sha256)
    return {spec.logo_sha256: data}


def save_watermark_defaults(
    current_settings: settings.AppSettings,
    spec: WatermarkSpec,
    assets: dict[str, bytes],
    *,
    settings_path: Path | None = None,
) -> None:
    """Commit the Logo before settings; keep the previous default on failure.

    Documents and their undo history own separate in-memory PNG bytes, so
    replacing the remembered Logo never removes a document's resources.
    """
    spec.validate_content()
    target = settings_path if settings_path is not None else settings.settings_file_path()
    created = None
    if spec.kind == "logo" and spec.logo_sha256:
        data = assets.get(spec.logo_sha256)
        if data is None:
            raise ValueError("无法记忆水印 Logo，请重新选择 Logo 图片")
        verify_logo(data, spec.logo_sha256)
        logo_path = _logo_path(spec, target)
        existed = logo_path.exists()
        # Replacing a corrupt remembered copy also repairs the same digest.
        atomic_write_bytes(logo_path, data)
        if not existed:
            created = logo_path
    try:
        settings.AppSettingsIO.save(replace(current_settings, last_watermark=spec), target)
    except (OSError, ValueError, TypeError):
        if created is not None:
            try:
                created.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    previous = current_settings.last_watermark
    if previous is not None and previous.logo_sha256 and previous.logo_sha256 != spec.logo_sha256:
        try:
            _logo_path(previous, target).unlink(missing_ok=True)
        except OSError:
            # An obsolete preferences copy can be left behind; the new
            # defaults are already committed and remain usable.
            pass
