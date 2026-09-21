"""Compare unchanged-image painting with the production watermark compositor.

Run with uv run --no-sync python scripts/benchmark_watermark.py --output PATH.
This measures Qt CPU rendering into a 1280x800 viewport, not desktop latency.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtCore import QPointF, QRectF, qVersion
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.models import ImageDocument
from fdm.ui.watermark_rendering import draw_watermark, import_logo, watermark_raster_cache
from fdm.watermark import WatermarkSpec


def summary(samples):
    return {"p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95))}


def run(iterations=60):
    app = QApplication.instance() or QApplication([])
    results = []
    with TemporaryDirectory(prefix="fdm-watermark-benchmark-") as folder:
        logo = QImage(180, 80, QImage.Format.Format_ARGB32)
        logo.fill(QColor(220, 40, 70, 160))
        path = Path(folder) / "logo.png"
        logo.save(str(path))
        digest, data = import_logo(path)
        for size in ((3840, 2160), (7680, 4320)):
            source = QImage(*size, QImage.Format.Format_RGB32)
            source.fill(QColor("#C5D5DF"))
            source_identity = source.cacheKey()
            document = ImageDocument(id="benchmark", path="synthetic.png", image_size=size)
            document.watermark_assets[digest] = data
            viewport = QImage(1280, 800, QImage.Format.Format_ARGB32_Premultiplied)
            for kind in ("text", "logo"):
                for layout in ("single", "tile"):
                    spec = WatermarkSpec(enabled=True, kind=kind, text="纤维测量 / Laboratory\n版权所有 © 2026", logo_sha256=digest, layout=layout, rotation=-30)
                    for gesture in ("pan", "zoom"):
                        watermark_raster_cache.clear()
                        timings = {}
                        warm_cache_bytes = peak_cache_bytes = 0
                        for enabled in (False, True):
                            document.watermark = replace(spec, enabled=enabled)
                            samples = []
                            for iteration in range(iterations + 12):
                                scale = 1000 / size[0]
                                if gesture == "zoom":
                                    scale *= (0.75, 1, 1.25, 1.5)[iteration % 4]
                                x, y = 45 - (iteration % 30) * 2, 30 - iteration % 20
                                start = time.perf_counter()
                                painter = QPainter(viewport)
                                painter.fillRect(viewport.rect(), QColor("#202830"))
                                painter.drawImage(QRectF(x, y, size[0] * scale, size[1] * scale), source)
                                draw_watermark(painter, document, lambda p: QPointF(x + p.x * scale, y + p.y * scale), strict=True)
                                painter.end()
                                elapsed = (time.perf_counter() - start) * 1000
                                if enabled:
                                    peak_cache_bytes = max(peak_cache_bytes, watermark_raster_cache.bytes)
                                    if iteration == 11:
                                        warm_cache_bytes = watermark_raster_cache.bytes
                                if iteration >= 12:
                                    samples.append(elapsed)
                            timings["watermark" if enabled else "baseline"] = summary(samples)
                        assert source.cacheKey() == source_identity
                        assert watermark_raster_cache.bytes <= watermark_raster_cache.budget
                        results.append({
                            "image_size": size, "kind": kind, "layout": layout, "gesture": gesture,
                            "iterations": iterations, **timings,
                            "p50_added_ms": timings["watermark"]["p50_ms"] - timings["baseline"]["p50_ms"],
                            "cache_bytes": watermark_raster_cache.bytes,
                            "cache_bytes_after_warmup": warm_cache_bytes,
                            "cache_peak_bytes": peak_cache_bytes,
                            "cache_growth_after_warmup_bytes": watermark_raster_cache.bytes - warm_cache_bytes,
                            "cache_hits": watermark_raster_cache.hits,
                            "cache_misses": watermark_raster_cache.misses,
                            "logo_decodes": watermark_raster_cache.logo_decodes,
                        })
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = peak if platform.system() == "Darwin" else peak * 1024
    except ImportError:
        peak_rss = None
    return {
        "platform": platform.platform(), "python": platform.python_version(), "qt": qVersion(),
        "viewport": [1280, 800], "cache_budget_bytes": watermark_raster_cache.budget,
        "process_peak_rss_bytes": peak_rss,
        "measurement": "Qt CPU offscreen repaint; synthetic source; not Windows installer acceptance",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.iterations < 30:
        parser.error("at least 30 measured iterations are required")
    report = run(options.iterations)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{len(report['results'])} scenarios written to {options.output}")
