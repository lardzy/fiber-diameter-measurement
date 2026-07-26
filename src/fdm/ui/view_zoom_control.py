from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QWheelEvent
from PySide6.QtWidgets import QMenu, QSizePolicy, QToolButton

from fdm.ui.view_transform import CanvasViewportSnapshot, CanvasZoomMode


class ViewZoomStatusButton(QToolButton):
    """Compact, explicit view-scale control for the status bar.

    The percentage is a display transform, not an optical microscope
    magnification.  Wheel events are consumed so merely scrolling across the
    status bar can never change the current view.
    """

    fitRequested = Signal()
    actualRequested = Signal()
    zoomRequested = Signal(float)
    customZoomRequested = Signal()

    _PRESET_ZOOMS = (0.25, 0.5, 2.0, 4.0)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("viewZoomStatusButton")
        self.setAutoRaise(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setText("视图缩放 —")
        self.setToolTip(
            "视图缩放只描述源图像像素与界面逻辑像素的显示比例，"
            "不代表显微镜光学倍率。\n"
            "100% = 1 个源图像像素对应 1 个 Qt 界面逻辑像素。"
        )
        self.setEnabled(False)
        self._last_display_signature: tuple[float, CanvasZoomMode, bool] | None = (
            None
        )

        menu = QMenu(self)
        fit_action = menu.addAction("适合窗口")
        fit_action.triggered.connect(self.fitRequested)
        actual_action = menu.addAction("100% · 原始像素")
        actual_action.triggered.connect(self.actualRequested)
        menu.addSeparator()

        self._preset_group = QActionGroup(menu)
        self._preset_group.setExclusive(False)
        for zoom in self._PRESET_ZOOMS:
            action = QAction(_format_percentage(zoom), menu)
            action.setData(zoom)
            action.triggered.connect(
                lambda _checked=False, value=zoom: self.zoomRequested.emit(value)
            )
            self._preset_group.addAction(action)
            menu.addAction(action)

        menu.addSeparator()
        custom_action = menu.addAction("自定义…")
        custom_action.triggered.connect(self.customZoomRequested)
        self.setMenu(menu)

    def set_viewport_snapshot(
        self,
        snapshot: CanvasViewportSnapshot | None,
        *,
        digital_slide: bool = False,
    ) -> None:
        self.setEnabled(snapshot is not None)
        if snapshot is None:
            self._last_display_signature = None
            self.setText("视图缩放 —")
            return
        signature = (
            float(snapshot.zoom),
            snapshot.mode,
            bool(digital_slide),
        )
        if signature == self._last_display_signature:
            return
        self._last_display_signature = signature

        percentage = _format_percentage(snapshot.zoom)
        if digital_slide:
            if snapshot.mode is CanvasZoomMode.FIT:
                label = f"视场适合 · {percentage}"
            elif snapshot.mode is CanvasZoomMode.ACTUAL:
                label = "视场原始像素 · 100%"
            else:
                label = f"视场缩放 · {percentage}"
        elif snapshot.mode is CanvasZoomMode.FIT:
            label = f"适合窗口 · {percentage}"
        elif snapshot.mode is CanvasZoomMode.ACTUAL:
            label = "原始像素 · 100%"
        else:
            label = f"视图缩放 · {percentage}"
        self.setText(label)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


def _format_percentage(zoom: float) -> str:
    try:
        percentage = float(zoom) * 100.0
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(percentage) or percentage <= 0:
        return "—"
    if percentage < 1.0:
        token = f"{percentage:.2f}"
    elif percentage < 100.0:
        token = f"{percentage:.1f}"
    elif math.isclose(percentage, round(percentage), abs_tol=1e-9):
        token = f"{percentage:.0f}"
    else:
        token = f"{percentage:.1f}"
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return f"{token}%"
