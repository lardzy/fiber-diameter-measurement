from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette, QShowEvent, QMoveEvent, QResizeEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from fdm.content_experiment import ContentExperimentRecord, ContentOverlayStyle, ContentRecordKind
from fdm.geometry import Line, Point


class MicroviewPreviewHost(QWidget):
    metricsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#101820"))
        self.setPalette(palette)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._content_overlay_style = ContentOverlayStyle.NONE
        self._content_overlay_records: list[ContentExperimentRecord] = []
        self._content_overlay_fiber_colors: dict[str, str] = {}
        self._content_overlay_pending_line: Line | None = None
        self._prefer_gdi_preview = False
        self.set_preview_resolution(640, 480)

    def ensure_native_handle(self) -> int:
        self.winId()
        return self.native_preview_handle()

    def set_preview_resolution(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        self.setFixedSize(width, height)
        self.metricsChanged.emit()

    def native_preview_handle(self) -> int:
        try:
            return int(self.winId())
        except Exception:
            return 0

    def native_preview_size(self) -> tuple[int, int]:
        size = self.size()
        return max(1, size.width()), max(1, size.height())

    def set_prefer_gdi_preview(self, enabled: bool) -> None:
        self._prefer_gdi_preview = bool(enabled)

    def prefer_gdi_preview(self) -> bool:
        return self._prefer_gdi_preview

    def set_content_experiment_overlay(
        self,
        *,
        overlay_style: str = ContentOverlayStyle.NONE,
        records: list[ContentExperimentRecord] | None = None,
        fiber_colors: dict[str, str] | None = None,
        pending_line: Line | None = None,
    ) -> None:
        self._content_overlay_style = overlay_style
        self._content_overlay_records = list(records or [])
        self._content_overlay_fiber_colors = dict(fiber_colors or {})
        self._content_overlay_pending_line = pending_line
        self.draw_content_experiment_overlay()

    def draw_content_experiment_overlay(self) -> None:
        if sys.platform != "win32":
            return
        hwnd = self.native_preview_handle()
        if hwnd <= 0:
            return
        width, height = self.native_preview_size()
        if width <= 1 or height <= 1:
            return
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hdc = user32.GetDC(hwnd)
        if not hdc:
            return
        try:
            gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
            self._draw_content_guides_gdi(gdi32, hdc, width, height)
            for record in self._content_overlay_records:
                if record.kind != ContentRecordKind.DIAMETER or record.line_px is None:
                    continue
                color = QColor(self._content_overlay_fiber_colors.get(record.fiber_id, "#7BD389"))
                self._draw_line_with_points_gdi(gdi32, hdc, record.line_px, color)
            if self._content_overlay_pending_line is not None:
                self._draw_line_with_points_gdi(gdi32, hdc, self._content_overlay_pending_line, QColor("#F4D35E"), points=False)
        finally:
            user32.ReleaseDC(hwnd, hdc)

    def _draw_content_guides_gdi(self, gdi32, hdc: int, width: int, height: int) -> None:
        center = Point(width / 2.0, height / 2.0)
        guide = QColor("#F4D35E")
        if self._content_overlay_style in {ContentOverlayStyle.HORIZONTAL, ContentOverlayStyle.CROSS, ContentOverlayStyle.CROSSHAIR}:
            self._draw_raw_line_gdi(gdi32, hdc, Point(0, center.y), Point(width, center.y), guide, 1)
        if self._content_overlay_style in {ContentOverlayStyle.VERTICAL, ContentOverlayStyle.CROSS, ContentOverlayStyle.CROSSHAIR}:
            self._draw_raw_line_gdi(gdi32, hdc, Point(center.x, 0), Point(center.x, height), guide, 1)
        if self._content_overlay_style in {ContentOverlayStyle.CENTER_DOT, ContentOverlayStyle.CROSSHAIR}:
            self._draw_raw_ellipse_gdi(gdi32, hdc, center, 5, guide)
        if self._content_overlay_style == ContentOverlayStyle.CROSSHAIR:
            self._draw_raw_ellipse_gdi(gdi32, hdc, center, 16, guide, fill=False)

    def _draw_line_with_points_gdi(self, gdi32, hdc: int, line: Line, color: QColor, *, points: bool = True) -> None:
        self._draw_raw_line_gdi(gdi32, hdc, line.start, line.end, QColor("#0B0B0B"), 4)
        self._draw_raw_line_gdi(gdi32, hdc, line.start, line.end, color, 2)
        if points:
            self._draw_raw_ellipse_gdi(gdi32, hdc, line.start, 4, color)
            self._draw_raw_ellipse_gdi(gdi32, hdc, line.end, 4, color)

    def _draw_raw_line_gdi(self, gdi32, hdc: int, start: Point, end: Point, color: QColor, width: int) -> None:
        pen = gdi32.CreatePen(0, max(1, int(width)), _colorref(color))
        old_pen = gdi32.SelectObject(hdc, pen)
        try:
            gdi32.MoveToEx(hdc, int(round(start.x)), int(round(start.y)), None)
            gdi32.LineTo(hdc, int(round(end.x)), int(round(end.y)))
        finally:
            gdi32.SelectObject(hdc, old_pen)
            gdi32.DeleteObject(pen)

    def _draw_raw_ellipse_gdi(self, gdi32, hdc: int, center: Point, radius: int, color: QColor, *, fill: bool = True) -> None:
        pen = gdi32.CreatePen(0, 1, _colorref(QColor("#0B0B0B")))
        brush = gdi32.CreateSolidBrush(_colorref(color)) if fill else gdi32.GetStockObject(5)  # NULL_BRUSH
        old_pen = gdi32.SelectObject(hdc, pen)
        old_brush = gdi32.SelectObject(hdc, brush)
        try:
            x = int(round(center.x))
            y = int(round(center.y))
            r = max(1, int(radius))
            gdi32.Ellipse(hdc, x - r, y - r, x + r + 1, y + r + 1)
        finally:
            gdi32.SelectObject(hdc, old_brush)
            gdi32.SelectObject(hdc, old_pen)
            gdi32.DeleteObject(pen)
            if fill:
                gdi32.DeleteObject(brush)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.ensure_native_handle()
        self.metricsChanged.emit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.metricsChanged.emit()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        self.metricsChanged.emit()


def _colorref(color: QColor) -> int:
    resolved = color if color.isValid() else QColor("#7BD389")
    return int(resolved.red()) | (int(resolved.green()) << 8) | (int(resolved.blue()) << 16)
