from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QStyle

try:
    import qtawesome as qta
except ImportError:  # pragma: no cover - optional dependency
    qta = None


QT_AWESOME_NAMES: dict[str, str] = {
    "open_images": "fa5s.images",
    "open_folder": "fa5s.folder-open",
    "open_project": "fa5s.file-import",
    "save_project": "fa5s.save",
    "close_current": "fa5s.window-close",
    "close_all": "fa5s.times-circle",
    "undo": "fa5s.undo",
    "redo": "fa5s.redo",
    "delete": "fa5s.trash-alt",
    "add": "fa5s.plus-circle",
    "rename": "fa5s.edit",
    "fit": "fa5s.expand-arrows-alt",
    "actual_size": "fa5s.expand",
    "export": "fa5s.file-export",
    "model": "fa5s.microchip",
    "preset_add": "fa5s.plus",
    "preset_apply": "fa5s.check",
    "select": "fa5s.mouse-pointer",
    "manual": "mdi6.vector-line",
    "snap": "mdi6.magnet",
    "polygon_area": "mdi6.draw-polygon",
    "freehand_area": "mdi6.draw",
    "calibration": "mdi6.ruler",
    "area_auto": "mdi6.image-filter-center-focus-strong",
}


def themed_icon(name: str, *, color: str = "#F7F4EA", size: int = 18) -> QIcon:
    if qta is not None:
        icon_name = QT_AWESOME_NAMES.get(name)
        if icon_name:
            try:
                return qta.icon(icon_name, color=color)
            except Exception:
                pass
    fallback = _FALLBACK_BUILDERS.get(name)
    if fallback is not None:
        return _paint_icon(fallback, color=color, size=size)
    standard = _STANDARD_ICONS.get(name)
    if standard is not None and QApplication.instance() is not None:
        return QApplication.style().standardIcon(standard)
    return QIcon()


def _paint_icon(builder: Callable[[QPainter, QColor, QRectF], None], *, color: str, size: int) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    builder(painter, QColor(color), QRectF(1.0, 1.0, size - 2.0, size - 2.0))
    painter.end()
    return QIcon(pixmap)


def _pen(color: QColor, width: float = 1.8) -> QPen:
    pen = QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_select(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.4))
    painter.setBrush(color)
    points = QPolygonF(
        [
            QPointF(rect.left() + rect.width() * 0.2, rect.top() + rect.height() * 0.12),
            QPointF(rect.left() + rect.width() * 0.76, rect.top() + rect.height() * 0.5),
            QPointF(rect.left() + rect.width() * 0.53, rect.top() + rect.height() * 0.55),
            QPointF(rect.left() + rect.width() * 0.68, rect.bottom() - rect.height() * 0.08),
            QPointF(rect.left() + rect.width() * 0.54, rect.bottom() - rect.height() * 0.02),
            QPointF(rect.left() + rect.width() * 0.4, rect.top() + rect.height() * 0.47),
            QPointF(rect.left() + rect.width() * 0.22, rect.bottom() - rect.height() * 0.08),
        ]
    )
    painter.drawPolygon(points)


def _draw_manual(painter: QPainter, color: QColor, rect: QRectF) -> None:
    start = QPointF(rect.left() + rect.width() * 0.18, rect.bottom() - rect.height() * 0.2)
    end = QPointF(rect.right() - rect.width() * 0.18, rect.top() + rect.height() * 0.22)
    painter.setPen(_pen(QColor("#0B0B0B"), 3.2))
    painter.drawLine(start, end)
    painter.setPen(_pen(color, 1.9))
    painter.drawLine(start, end)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#0B0B0B"))
    painter.drawEllipse(start, 2.8, 2.8)
    painter.drawEllipse(end, 2.8, 2.8)
    painter.setBrush(color)
    painter.drawEllipse(start, 1.7, 1.7)
    painter.drawEllipse(end, 1.7, 1.7)


def _draw_snap(painter: QPainter, color: QColor, rect: QRectF) -> None:
    _draw_manual(painter, color, rect)
    center = QPointF(rect.center().x(), rect.top() + rect.height() * 0.32)
    painter.setPen(_pen(color, 1.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(center, 3.0, 3.0)
    painter.drawLine(QPointF(center.x() - 4.6, center.y()), QPointF(center.x() + 4.6, center.y()))
    painter.drawLine(QPointF(center.x(), center.y() - 4.6), QPointF(center.x(), center.y() + 4.6))


def _draw_calibration(painter: QPainter, color: QColor, rect: QRectF) -> None:
    ruler = QRectF(rect.left() + rect.width() * 0.15, rect.top() + rect.height() * 0.38, rect.width() * 0.7, rect.height() * 0.24)
    painter.setPen(_pen(color, 1.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(ruler, 2.0, 2.0)
    for factor, height_factor in [(0.18, 0.16), (0.34, 0.22), (0.50, 0.16), (0.66, 0.22), (0.82, 0.16)]:
        x = ruler.left() + ruler.width() * factor
        tick_height = rect.height() * height_factor
        painter.drawLine(QPointF(x, ruler.top()), QPointF(x, ruler.top() + tick_height))


def _draw_polygon_area(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.5))
    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70))
    polygon = QPolygonF(
        [
            QPointF(rect.left() + rect.width() * 0.18, rect.bottom() - rect.height() * 0.18),
            QPointF(rect.left() + rect.width() * 0.33, rect.top() + rect.height() * 0.24),
            QPointF(rect.right() - rect.width() * 0.18, rect.top() + rect.height() * 0.34),
            QPointF(rect.right() - rect.width() * 0.28, rect.bottom() - rect.height() * 0.2),
        ]
    )
    painter.drawPolygon(polygon)
    painter.setBrush(color)
    for point in polygon:
        painter.drawEllipse(point, 1.7, 1.7)


def _draw_freehand_area(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.7))
    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 52))
    path = QPainterPath(QPointF(rect.left() + rect.width() * 0.15, rect.center().y()))
    path.cubicTo(
        QPointF(rect.left() + rect.width() * 0.28, rect.top() + rect.height() * 0.12),
        QPointF(rect.left() + rect.width() * 0.62, rect.top() + rect.height() * 0.18),
        QPointF(rect.right() - rect.width() * 0.18, rect.center().y() - rect.height() * 0.08),
    )
    path.cubicTo(
        QPointF(rect.right() - rect.width() * 0.12, rect.bottom() - rect.height() * 0.18),
        QPointF(rect.left() + rect.width() * 0.42, rect.bottom() - rect.height() * 0.1),
        QPointF(rect.left() + rect.width() * 0.15, rect.center().y()),
    )
    painter.drawPath(path)


def _draw_area_auto(painter: QPainter, color: QColor, rect: QRectF) -> None:
    _draw_polygon_area(painter, color, rect)
    painter.setPen(_pen(QColor("#0B0B0B"), 1.6))
    painter.drawLine(QPointF(rect.center().x(), rect.top() + 3.0), QPointF(rect.center().x(), rect.top() + 8.0))
    painter.drawLine(QPointF(rect.center().x() - 2.5, rect.top() + 5.5), QPointF(rect.center().x() + 2.5, rect.top() + 5.5))


def _draw_images(painter: QPainter, color: QColor, rect: QRectF) -> None:
    back = QRectF(rect.left() + 2.0, rect.top() + 4.0, rect.width() * 0.62, rect.height() * 0.62)
    front = QRectF(rect.left() + 6.0, rect.top() + 7.0, rect.width() * 0.62, rect.height() * 0.62)
    painter.setPen(_pen(color, 1.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(back, 2.0, 2.0)
    painter.drawRoundedRect(front, 2.0, 2.0)
    path = QPainterPath(QPointF(front.left() + 2.5, front.bottom() - 3.0))
    path.lineTo(front.left() + 7.0, front.top() + front.height() * 0.58)
    path.lineTo(front.left() + 10.5, front.top() + front.height() * 0.74)
    path.lineTo(front.right() - 2.0, front.top() + front.height() * 0.38)
    painter.drawPath(path)
    painter.drawEllipse(QPointF(front.right() - 5.0, front.top() + 4.5), 1.3, 1.3)


def _draw_export(painter: QPainter, color: QColor, rect: QRectF) -> None:
    box = QRectF(rect.left() + 3.0, rect.top() + rect.height() * 0.46, rect.width() * 0.62, rect.height() * 0.34)
    painter.setPen(_pen(color, 1.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(box, 2.0, 2.0)
    painter.drawLine(QPointF(rect.center().x(), rect.top() + 3.0), QPointF(rect.center().x(), box.top() + 1.0))
    painter.drawLine(QPointF(rect.center().x(), rect.top() + 3.0), QPointF(rect.center().x() - 3.4, rect.top() + 6.6))
    painter.drawLine(QPointF(rect.center().x(), rect.top() + 3.0), QPointF(rect.center().x() + 3.4, rect.top() + 6.6))


def _draw_fit(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.6))
    margin = 3.0
    corner = 4.0
    left = rect.left() + margin
    top = rect.top() + margin
    right = rect.right() - margin
    bottom = rect.bottom() - margin
    painter.drawLine(QPointF(left, top + corner), QPointF(left, top))
    painter.drawLine(QPointF(left, top), QPointF(left + corner, top))
    painter.drawLine(QPointF(right - corner, top), QPointF(right, top))
    painter.drawLine(QPointF(right, top), QPointF(right, top + corner))
    painter.drawLine(QPointF(left, bottom - corner), QPointF(left, bottom))
    painter.drawLine(QPointF(left, bottom), QPointF(left + corner, bottom))
    painter.drawLine(QPointF(right - corner, bottom), QPointF(right, bottom))
    painter.drawLine(QPointF(right, bottom - corner), QPointF(right, bottom))


def _draw_actual_size(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.5))
    box = QRectF(rect.left() + 3.5, rect.top() + 3.5, rect.width() - 7.0, rect.height() - 7.0)
    painter.drawRect(box)
    center = rect.center()
    painter.drawLine(QPointF(center.x(), box.top() + 2.0), QPointF(center.x(), box.bottom() - 2.0))
    painter.drawLine(QPointF(box.left() + 2.0, center.y()), QPointF(box.right() - 2.0, center.y()))


def _draw_add(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.8))
    painter.drawEllipse(rect.adjusted(3.0, 3.0, -3.0, -3.0))
    painter.drawLine(QPointF(rect.center().x(), rect.top() + 5.0), QPointF(rect.center().x(), rect.bottom() - 5.0))
    painter.drawLine(QPointF(rect.left() + 5.0, rect.center().y()), QPointF(rect.right() - 5.0, rect.center().y()))


def _draw_rename(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.6))
    painter.drawLine(QPointF(rect.left() + 4.0, rect.bottom() - 4.0), QPointF(rect.right() - 5.0, rect.top() + 5.0))
    painter.drawLine(QPointF(rect.left() + 5.0, rect.bottom() - 6.0), QPointF(rect.left() + 7.0, rect.bottom() - 2.0))
    painter.drawLine(QPointF(rect.left() + 7.0, rect.bottom() - 2.0), QPointF(rect.left() + 3.0, rect.bottom()))


def _draw_model(painter: QPainter, color: QColor, rect: QRectF) -> None:
    chip = QRectF(rect.left() + 4.5, rect.top() + 4.5, rect.width() - 9.0, rect.height() - 9.0)
    painter.setPen(_pen(color, 1.4))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(chip, 2.0, 2.0)
    for step in (0.18, 0.38, 0.62, 0.82):
        x = chip.left() + chip.width() * step
        painter.drawLine(QPointF(x, rect.top() + 1.5), QPointF(x, chip.top()))
        painter.drawLine(QPointF(x, chip.bottom()), QPointF(x, rect.bottom() - 1.5))
        y = chip.top() + chip.height() * step
        painter.drawLine(QPointF(rect.left() + 1.5, y), QPointF(chip.left(), y))
        painter.drawLine(QPointF(chip.right(), y), QPointF(rect.right() - 1.5, y))


def _draw_apply(painter: QPainter, color: QColor, rect: QRectF) -> None:
    painter.setPen(_pen(color, 1.8))
    painter.drawLine(QPointF(rect.left() + 4.0, rect.center().y()), QPointF(rect.left() + 8.0, rect.bottom() - 5.0))
    painter.drawLine(QPointF(rect.left() + 8.0, rect.bottom() - 5.0), QPointF(rect.right() - 4.0, rect.top() + 5.0))


_FALLBACK_BUILDERS: dict[str, Callable[[QPainter, QColor, QRectF], None]] = {
    "select": _draw_select,
    "manual": _draw_manual,
    "snap": _draw_snap,
    "polygon_area": _draw_polygon_area,
    "freehand_area": _draw_freehand_area,
    "calibration": _draw_calibration,
    "area_auto": _draw_area_auto,
    "open_images": _draw_images,
    "export": _draw_export,
    "fit": _draw_fit,
    "actual_size": _draw_actual_size,
    "add": _draw_add,
    "rename": _draw_rename,
    "model": _draw_model,
    "preset_add": _draw_add,
    "preset_apply": _draw_apply,
}


_STANDARD_ICONS: dict[str, QStyle.StandardPixmap] = {
    "open_folder": QStyle.StandardPixmap.SP_DirOpenIcon,
    "open_project": QStyle.StandardPixmap.SP_FileIcon,
    "save_project": QStyle.StandardPixmap.SP_DialogSaveButton,
    "close_current": QStyle.StandardPixmap.SP_DialogCloseButton,
    "close_all": QStyle.StandardPixmap.SP_DialogCloseButton,
    "undo": QStyle.StandardPixmap.SP_ArrowBack,
    "redo": QStyle.StandardPixmap.SP_ArrowForward,
    "delete": QStyle.StandardPixmap.SP_TrashIcon,
}
