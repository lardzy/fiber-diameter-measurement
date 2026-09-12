"""Conservative whole-geometry quality checks, independent of image decoding."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainterPath, QPainterPathStroker

from fdm.services.slide_layout import SlideLayoutSnapshot
from fdm.services.slide_raster import SlideRasterSource


def _geometry(measurement):
    path = QPainterPath()
    if measurement.measurement_kind == "area":
        path.setFillRule(Qt.FillRule.OddEvenFill)
        for ring in measurement.area_rings_px or [measurement.polygon_px]:
            if len(ring) < 3:
                continue
            path.moveTo(ring[0].x, ring[0].y)
            for point in ring[1:]:
                path.lineTo(point.x, point.y)
            path.closeSubpath()
    elif measurement.point_px is not None:
        path.addEllipse(QRectF(measurement.point_px.x - .5, measurement.point_px.y - .5, 1, 1))
    else:
        points = measurement.polyline_px
        if not points and measurement.line_px is not None:
            line = measurement.effective_line()
            points = [line.start, line.end]
        if points:
            path.moveTo(points[0].x, points[0].y)
            for point in points[1:]:
                path.lineTo(point.x, point.y)
            stroker = QPainterPathStroker()
            stroker.setWidth(1)
            path = stroker.createStroke(path)
    return path


def annotate_measurement(document, measurement):
    payload = document.metadata.get("stitch_layout")
    if not isinstance(payload, dict):
        return
    layout_id = payload.get("layout_id")
    runtime = document._stitch_quality_runtime
    if runtime is None or runtime[0] != layout_id:
        layout = SlideLayoutSnapshot.from_dict(payload)
        runtime = (layout_id, SlideRasterSource(layout, lambda tile: None), {}, {})
        document._stitch_quality_runtime = runtime
    _, source, planes, checked = runtime
    segmentation = measurement.debug_payload.get("segmentation_source", {})
    context = measurement.source_context
    focus = context.get("focus_index", segmentation.get("focus_index"))
    if focus is None:
        focus = document.metadata.get("digital_slide", {}).get("focus_index", 0)
    focus = int(focus or 0)
    key = (id(measurement), measurement.geometry_revision, focus)
    if checked.get(measurement.id) == key:
        return
    if focus not in planes:
        coverage = QPainterPath()
        coverage.setFillRule(Qt.FillRule.WindingFill)
        for tile in source._planes.get(focus, ()):
            coverage.addRect(QRectF(tile.x, tile.y, tile.width, tile.height))
        bands = [QRectF(*rect) for rect in source.unverified_regions(focus)]
        planes[focus] = coverage.simplified(), bands
    coverage, bands = planes[focus]
    geometry = _geometry(measurement)
    unverified = any(geometry.intersects(rect) or geometry.contains(rect) for rect in bands)
    missing = not geometry.isEmpty() and not geometry.subtracted(coverage).isEmpty()
    truncated = bool(segmentation.get("seam_truncated"))
    measurement.source_context = {
        "layout_id": layout_id, "source_digest": source.layout.source_digest,
        "focus_index": focus, "sampling": source.layout.sampling,
        "quality": "incomplete" if missing or truncated else "unverified_seam" if unverified else "verified_region",
        "count_deduplicated": False,
    }
    checked[measurement.id] = key
    # Bound runtime state when objects are deleted or undo creates replacements.
    if len(checked) > len(document.measurements) + 64:
        live = {item.id for item in document.measurements}
        for item_id in tuple(checked):
            if item_id not in live:
                checked.pop(item_id)


def quality_label(measurement):
    return {"incomplete": "不完整 · 接缝／覆盖未验证", "unverified_seam": "接缝未验证"}.get(measurement.source_context.get("quality"), "")


def quality_rows(documents):
    rows = []
    for document in documents:
        if not document.metadata.get("stitch_layout"):
            continue
        for measurement in document.measurements:
            annotate_measurement(document, measurement)
            context = measurement.source_context
            rows.append({"图片": Path(document.path).name, "对象 ID": measurement.id,
                         "布局 ID": context.get("layout_id"), "源摘要": context.get("source_digest"),
                         "焦层": int(context.get("focus_index", 0)) + 1,
                         "拼接质量": quality_label(measurement) or "几何范围未跨未验证接缝",
                         "说明": "计数未经自动去重；原始视场保留。" if measurement.measurement_kind == "count" else "二维平移；单源线性采样。"})
    return rows
