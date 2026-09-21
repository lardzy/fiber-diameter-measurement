"""Small production-renderer and bundled image-codec probe."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QPointF, qVersion
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QImage, QPainter

from fdm.models import ImageDocument
from fdm.services.watermark_preferences import load_watermark_default_assets, save_watermark_defaults
from fdm.settings import AppSettings, AppSettingsIO
from fdm.ui.qt_raster_runtime import ensure_raster_application
from fdm.ui.watermark_rendering import draw_watermark, import_logo, watermark_geometry, watermark_raster_cache
from fdm.watermark import WatermarkSpec

_application = None


def run_watermark_self_check() -> dict:
    global _application
    _application = ensure_raster_application("watermark-self-check")
    font_count = len(QFontDatabase.families())
    runtime = {
        "qt_platform": _application.platformName(),
        "qt_version": qVersion(),
        "font_family_count": font_count,
        "default_font_family": QFontInfo(QFont()).family(),
    }
    checks = {"font_database": font_count > 0}
    details = {}
    document = ImageDocument(id="watermark-probe", path="probe.png", image_size=(320, 240))

    def render(spec, dpr=1.0):
        document.watermark = spec
        image = QImage(round(320 * dpr), round(240 * dpr), QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        image.fill(QColor("white"))
        painter = QPainter(image)
        try:
            draw_watermark(painter, document, lambda p: QPointF(p.x, p.y), strict=True)
        finally:
            painter.end()
        return image

    with TemporaryDirectory(prefix="fdm-watermark-probe-") as temporary:
        logo = QImage(32, 16, QImage.Format.Format_ARGB32)
        logo.fill(QColor(255, 0, 0, 128))
        for suffix in ("png", "jpg", "webp"):
            path = Path(temporary) / f"水印-logo.{suffix}"
            if not logo.save(str(path)):
                raise RuntimeError(f"watermark {suffix} writer unavailable")
            digest, data = import_logo(path)
            checks[f"codec_{suffix}"] = True
            if suffix == "png":
                document.watermark_assets[digest] = data
                logo_spec = WatermarkSpec(enabled=True, kind="logo", logo_sha256=digest, anchor="top_left", offset_x=0, offset_y=0)
        for dpr in (1.0, 1.5, 2.0):
            image = render(logo_spec, dpr)
            color = image.pixelColor(round(10 * dpr), round(10 * dpr))
            checks[f"logo_alpha@{dpr:g}"] = color.red() == 255 and abs(color.green() - 223) <= 1 and color.alpha() == 255
            details[f"logo_alpha@{dpr:g}"] = {"rgba": color.getRgb()}
            text = WatermarkSpec(enabled=True, text="纤维测量\nWatermark", color="#000000")
            blank = render(None, dpr)
            checks[f"text@{dpr:g}"] = render(text, dpr) != blank
            details[f"text@{dpr:g}"] = {"visible": checks[f"text@{dpr:g}"]}
            tiled = replace(logo_spec, layout="tile")
            result = render(tiled, dpr)
            checks[f"tile@{dpr:g}"] = result.pixelColor(round(130 * dpr), round(70 * dpr)) == color
            hits = watermark_raster_cache.hits
            checks[f"cache@{dpr:g}"] = render(tiled, dpr) == result and watermark_raster_cache.hits > hits
            for kind, base in (("text", text), ("logo", logo_spec)):
                dated = replace(
                    base, include_datetime=True, datetime_text="2026-09-21 14:35:26",
                    anchor="top_left", offset_x=0, offset_y=0, width_ratio=0.75,
                )
                geometry = watermark_geometry(document, dated)
                dated_image = render(dated, dpr)
                caption_box = (
                    0, math.floor(geometry.datetime_top * dpr),
                    math.ceil(geometry.width * dpr),
                    math.ceil((geometry.height - geometry.datetime_top) * dpr),
                )
                visible = dated_image.copy(*caption_box) != blank.copy(*caption_box)
                repeat_equal = render(dated, dpr) == dated_image
                checks[f"datetime_{kind}@{dpr:g}"] = visible and repeat_equal
                details[f"datetime_{kind}@{dpr:g}"] = {
                    "visible": visible, "repeat_equal": repeat_equal, "caption_box": caption_box,
                }
        settings_path = Path(temporary) / "profile" / "settings.json"
        remembered = replace(logo_spec, include_datetime=True, datetime_text="2026-09-21 14:35:26")
        save_watermark_defaults(AppSettings(), remembered, document.watermark_assets, settings_path=settings_path)
        restored = AppSettingsIO.load(settings_path)
        checks["preferences_roundtrip"] = (
            restored.last_watermark == remembered
            and load_watermark_default_assets(restored.last_watermark, settings_path=settings_path) == document.watermark_assets
        )
        document.document_kind = "digital_slide"
        checks["digital_slide_excluded"] = render(
            replace(logo_spec, include_datetime=True, datetime_text="2026-09-21 14:35:26")
        ) == render(None)
    return {
        "ok": all(checks.values()), "cases": checks,
        "failed_cases": [name for name, passed in checks.items() if not passed],
        "runtime": runtime, "details": details,
        "cache_bytes": watermark_raster_cache.bytes, "cache_budget": watermark_raster_cache.budget,
    }


if __name__ == "__main__":
    report = run_watermark_self_check()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if report["ok"] else 1)
