"""Small production-renderer and bundled image-codec probe."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter

from fdm.models import ImageDocument
from fdm.ui.watermark_rendering import draw_watermark, import_logo, watermark_raster_cache
from fdm.watermark import WatermarkSpec

_application = None


def run_watermark_self_check() -> dict:
    global _application
    _application = QGuiApplication.instance() or QGuiApplication(["watermark-self-check", "-platform", "offscreen"])
    checks = {}
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
            text = WatermarkSpec(enabled=True, text="纤维测量\nWatermark", color="#000000")
            blank = render(None, dpr)
            checks[f"text@{dpr:g}"] = render(text, dpr) != blank
            tiled = replace(logo_spec, layout="tile")
            result = render(tiled, dpr)
            checks[f"tile@{dpr:g}"] = result.pixelColor(round(130 * dpr), round(70 * dpr)) == color
            hits = watermark_raster_cache.hits
            checks[f"cache@{dpr:g}"] = render(tiled, dpr) == result and watermark_raster_cache.hits > hits
        document.document_kind = "digital_slide"
        checks["digital_slide_excluded"] = render(logo_spec) == render(None)
    return {"ok": all(checks.values()), "cases": checks, "cache_bytes": watermark_raster_cache.bytes, "cache_budget": watermark_raster_cache.budget}


if __name__ == "__main__":
    report = run_watermark_self_check()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if report["ok"] else 1)
