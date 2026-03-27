from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette, QShowEvent, QMoveEvent, QResizeEvent
from PySide6.QtWidgets import QSizePolicy, QWidget


class MicroviewPreviewHost(QWidget):
    metricsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#101820"))
        self.setPalette(palette)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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
