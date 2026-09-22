"""Synthetic Qt CPU scale preview benchmark; not Windows desktop acceptance.

uv run --no-sync python scripts/benchmark_scale_overlay.py --output report.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtCore import QRectF, qVersion
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from fdm.scale_overlay import ScaleOverlaySpec
from fdm.ui.scale_overlay_rendering import (
    layout_scale_overlay,
    paint_scale_overlay,
)


def percentile(samples):
    return {
        "p50_ms": round(float(np.percentile(samples, 50)), 4),
        "p95_ms": round(float(np.percentile(samples, 95)), 4),
    }


def peak_rss():
    try:
        import resource

        size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return size if sys.platform == "darwin" else size * 1024
    except ImportError:
        return None


def run(iterations=120):
    _app = QApplication.instance() or QApplication([])
    viewport = QImage(1280, 800, QImage.Format.Format_ARGB32_Premultiplied)
    results = []
    for width, height in ((3840, 2160), (7680, 4320)):
        sources = [QImage(width, height, QImage.Format.Format_RGB32) for _ in range(2)]
        for source, color in zip(sources, ("#203040", "#405060")):
            source.fill(QColor(color))
        identities = [source.cacheKey() for source in sources]
        for region in ("image", "viewport"):
            for gesture in ("drag", "pan", "zoom", "switch_image"):
                layout_scale_overlay.cache_clear()
                tracemalloc.start()
                timing = {}
                warm_memory = warm_rss = None
                spec = ScaleOverlaySpec(preview_region=region)
                for enabled in (False, True):
                    samples = []
                    for index in range(iterations + 30):
                        # Twenty repeated positions/zooms fit the bounded cache;
                        # long unique drags are measured separately below.
                        phase = index % 20
                        zoom = 1100 / width
                        if gesture == "zoom":
                            zoom *= 1 + phase / 40
                        dx = 20 - phase * 4 if gesture == "pan" else 20
                        dy = 30 - phase * 2 if gesture == "pan" else 30
                        source = sources[index % 2 if gesture == "switch_image" else 0]
                        started = time.perf_counter()
                        painter = QPainter(viewport)
                        try:
                            painter.fillRect(viewport.rect(), QColor("#172028"))
                            painter.drawImage(
                                QRectF(dx, dy, width * zoom, height * zoom), source
                            )
                            if enabled:
                                target = (0.0, 0.0, float(width), float(height))
                                if region == "viewport":
                                    left, top = (
                                        max(0.0, -dx / zoom),
                                        max(0.0, -dy / zoom),
                                    )
                                    target = (
                                        left,
                                        top,
                                        min(width, (1280 - dx) / zoom) - left,
                                        min(height, (800 - dy) / zoom) - top,
                                    )
                                current = (
                                    replace(
                                        spec, position="manual", relative_x=phase / 20
                                    )
                                    if gesture == "drag"
                                    else spec
                                )
                                layout = layout_scale_overlay(
                                    current, target, 4.0, "um"
                                )
                                painter.translate(dx, dy)
                                painter.scale(zoom, zoom)
                                paint_scale_overlay(painter, layout)
                        finally:
                            painter.end()
                        if index >= 30:
                            samples.append((time.perf_counter() - started) * 1000)
                        if enabled and index == 29:
                            warm_memory = tracemalloc.get_traced_memory()[0]
                            warm_rss = peak_rss()
                    timing["scale_preview" if enabled else "image_only"] = percentile(
                        samples
                    )
                python_bytes, peak_python = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                assert [source.cacheKey() for source in sources] == identities
                info = layout_scale_overlay.cache_info()
                results.append(
                    {
                        "size": [width, height],
                        "region": region,
                        "gesture": gesture,
                        "iterations": iterations,
                        **timing,
                        "cache": info._asdict(),
                        "python_growth_after_warmup": python_bytes - warm_memory,
                        "python_peak_bytes": peak_python,
                        "process_peak_rss_after_warmup": warm_rss,
                        "process_peak_rss_after_run": peak_rss(),
                        "source_unchanged": True,
                    }
                )
    # Exercise eviction, not just repeated cache hits.
    layout_scale_overlay.cache_clear()
    tracemalloc.start()
    checkpoints = []
    for index in range(2048):
        spec = ScaleOverlaySpec(position="manual", relative_x=(index + 1) / 2049)
        layout_scale_overlay(spec, (0.0, 0.0, 7680.0, 4320.0), 4.0, "um")
        if index in (511, 1023, 1535, 2047):
            checkpoints.append(
                {
                    "layouts": index + 1,
                    "python_bytes": tracemalloc.get_traced_memory()[0],
                    "cached_entries": layout_scale_overlay.cache_info().currsize,
                }
            )
    tracemalloc.stop()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "qt": qVersion(),
        "measurement": "Synthetic Qt CPU raster into 1280x800 viewport; excludes desktop input latency and Windows installation",
        "results": results,
        "unique_drag_memory": checkpoints,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.iterations < 30:
        parser.error("at least 30 measured iterations required")
    report = run(options.iterations)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(report['results'])} scenarios written to {options.output}")
