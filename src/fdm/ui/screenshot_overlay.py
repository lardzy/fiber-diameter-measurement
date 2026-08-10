from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QKeyEvent, QMouseEvent, QPaintEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from fdm.services.screenshot_capture import (
    CaptureRect,
    CaptureSelection,
    ScreenInfo,
    WindowCandidate,
    candidate_at_point,
    union_rect,
)


def logical_point_to_physical(point: QPoint, screens: Sequence[ScreenInfo]) -> QPoint:
    """Map a Qt global logical point to native desktop pixels.

    Global origins are never multiplied by DPR.  Only the offset within the
    owning monitor is scaled, which is essential for left/top negative screens.
    """

    screen = next(
        (
            item
            for item in screens
            if item.logical_rect.contains(point.x(), point.y())
        ),
        None,
    )
    if screen is None:
        screen = next((item for item in screens if item.primary), screens[0] if screens else None)
    if screen is None:
        return QPoint(point)
    logical = screen.logical_rect
    physical = screen.physical_rect
    ratio_x = physical.width / max(1, logical.width)
    ratio_y = physical.height / max(1, logical.height)
    return QPoint(
        physical.x + round((point.x() - logical.x) * ratio_x),
        physical.y + round((point.y() - logical.y) * ratio_y),
    )


def logical_rect_to_physical(rect: CaptureRect, screens: Sequence[ScreenInfo]) -> CaptureRect:
    fragments = [
        screen.logical_fragment_to_physical(rect)
        for screen in screens
        if rect.intersection(screen.logical_rect) is not None
    ]
    mapped = union_rect(item for item in fragments if item.valid)
    if mapped is not None:
        return mapped
    first = logical_point_to_physical(QPoint(rect.x, rect.y), screens)
    last = logical_point_to_physical(QPoint(rect.right, rect.bottom), screens)
    return CaptureRect(first.x(), first.y(), last.x() - first.x(), last.y() - first.y()).normalized()


def physical_rect_to_logical(rect: CaptureRect, screens: Sequence[ScreenInfo]) -> CaptureRect:
    fragments = [
        screen.physical_fragment_to_logical(rect)
        for screen in screens
        if rect.intersection(screen.physical_rect) is not None
    ]
    mapped = union_rect(item for item in fragments if item.valid)
    return mapped or CaptureRect(rect.x, rect.y, rect.width, rect.height)


class ScreenshotOverlay(QWidget):
    """Virtual-desktop selection overlay with nested Win32 candidate cycling."""

    selectionAccepted = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        screens: Sequence[ScreenInfo],
        candidates: Sequence[WindowCandidate] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._screens = tuple(screens)
        self._candidates = tuple(candidates)
        self._hover_candidates: tuple[WindowCandidate, ...] = ()
        self._candidate_index = 0
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._selection: CaptureSelection | None = None
        self._accepted = False

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        desktop = union_rect(screen.logical_rect for screen in self._screens)
        if desktop is not None:
            self.setGeometry(desktop.to_qrect())

    @property
    def screens(self) -> tuple[ScreenInfo, ...]:
        return self._screens

    @property
    def current_selection(self) -> CaptureSelection | None:
        return self._selection

    @property
    def hover_candidates(self) -> tuple[WindowCandidate, ...]:
        return self._hover_candidates

    @property
    def selected_candidate(self) -> WindowCandidate | None:
        if not self._hover_candidates:
            return None
        return self._hover_candidates[self._candidate_index % len(self._hover_candidates)]

    def set_candidates(self, candidates: Sequence[WindowCandidate]) -> None:
        self._candidates = tuple(candidates)
        self._refresh_hover(self.mapFromGlobal(self.cursor().pos()))

    def begin(self) -> None:
        self._accepted = False
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        # A screenshot hotkey commonly leaves the pointer completely still.
        # Prime smart-window recognition at the existing cursor position so a
        # direct click accepts the candidate instead of producing an empty
        # zero-size region on the first attempt.
        self._refresh_hover(self.mapFromGlobal(self.cursor().pos()))

    def cycle_candidate(self, step: int = 1) -> WindowCandidate | None:
        if not self._hover_candidates:
            return None
        self._candidate_index = (self._candidate_index + int(step)) % len(self._hover_candidates)
        self.update()
        return self.selected_candidate

    def accept_candidate(self) -> bool:
        candidate = self.selected_candidate
        if candidate is None:
            return False
        self._accept(CaptureSelection(candidate.capture_rect, candidate=candidate))
        return True

    def accept_region(self, rect: CaptureRect) -> bool:
        normalized = rect.normalized()
        if not normalized.valid:
            return False
        self._accept(CaptureSelection(normalized))
        return True

    def cancel(self) -> None:
        if self._accepted:
            return
        self._drag_origin = None
        self._drag_current = None
        self._selection = None
        self.hide()
        self.cancelled.emit()

    def _accept(self, selection: CaptureSelection) -> None:
        if self._accepted:
            return
        self._accepted = True
        self._selection = selection
        self.hide()
        self.selectionAccepted.emit(selection)

    def _global_logical(self, local: QPoint) -> QPoint:
        return self.mapToGlobal(local)

    def _physical_point(self, local: QPoint) -> QPoint:
        return logical_point_to_physical(self._global_logical(local), self._screens)

    def _refresh_hover(self, local: QPoint) -> None:
        if self._drag_origin is not None:
            return
        hits = candidate_at_point(self._candidates, self._physical_point(local))
        handles = tuple(item.handle for item in hits)
        old_handles = tuple(item.handle for item in self._hover_candidates)
        if handles != old_handles:
            self._hover_candidates = hits
            self._candidate_index = 0
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        local = event.position().toPoint()
        if self._drag_origin is not None:
            self._drag_current = local
            self.update()
        else:
            self._refresh_hover(local)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() is Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self._drag_current = self._drag_origin
            event.accept()
            return
        if event.button() is Qt.MouseButton.RightButton:
            self.cancel()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() is not Qt.MouseButton.LeftButton or self._drag_origin is None:
            super().mouseReleaseEvent(event)
            return
        end = event.position().toPoint()
        start = self._drag_origin
        self._drag_origin = None
        self._drag_current = None
        if (end - start).manhattanLength() <= 4 and self.accept_candidate():
            event.accept()
            return
        global_start = self._global_logical(start)
        global_end = self._global_logical(end)
        logical = CaptureRect(
            global_start.x(),
            global_start.y(),
            global_end.x() - global_start.x(),
            global_end.y() - global_start.y(),
        ).normalized()
        self.accept_region(logical_rect_to_physical(logical, self._screens))
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        if delta:
            self.cycle_candidate(1 if delta < 0 else -1)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.cancel()
        elif key in (Qt.Key.Key_Tab, Qt.Key.Key_PageDown):
            self.cycle_candidate(1)
        elif key in (Qt.Key.Key_Backtab, Qt.Key.Key_PageUp):
            self.cycle_candidate(-1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.accept_candidate()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 92))

        highlight: CaptureRect | None = None
        label = "拖动选择区域；Esc 取消"
        if self._drag_origin is not None and self._drag_current is not None:
            local = QRect(self._drag_origin, self._drag_current).normalized()
            highlight = CaptureRect.from_qrect(local)
            label = f"{local.width()} × {local.height()} DIP"
        elif (candidate := self.selected_candidate) is not None:
            logical = physical_rect_to_logical(candidate.capture_rect, self._screens)
            global_origin = self.geometry().topLeft()
            highlight = logical.translated(-global_origin.x(), -global_origin.y())
            title = candidate.title.strip() or candidate.class_name.strip() or "窗口区域"
            label = (
                "单击确认；拖动自由框选；Tab/滚轮切换层级  ·  "
                f"{title}  {candidate.capture_rect.width} × {candidate.capture_rect.height} px"
            )

        if highlight is not None and highlight.valid:
            rect = highlight.to_qrect()
            # A fully transparent pixel in a Windows layered window is also
            # excluded from native hit testing.  Clearing the recognised area
            # therefore made clicks pass through to the target application:
            # the shaded taskbar/desktop edge worked, while the visually clear
            # window area appeared unresponsive.  Alpha 1 is visually clear
            # but keeps the overlay as the mouse target.
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(rect, QColor(0, 0, 0, 1))
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(45, 180, 255), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
            anchor = rect.bottomLeft() + QPoint(0, 8)
        else:
            anchor = QPoint(16, 16)

        metrics = QFontMetrics(painter.font())
        text_rect = metrics.boundingRect(label).adjusted(-8, -5, 8, 5)
        text_rect.moveTopLeft(anchor)
        if text_rect.right() > self.width() - 8:
            text_rect.moveRight(self.width() - 8)
        if text_rect.bottom() > self.height() - 8:
            text_rect.moveBottom(max(8, anchor.y() - 8))
        painter.fillRect(text_rect, QColor(24, 24, 24, 220))
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)


__all__ = [
    "ScreenshotOverlay",
    "logical_point_to_physical",
    "logical_rect_to_physical",
    "physical_rect_to_logical",
]
