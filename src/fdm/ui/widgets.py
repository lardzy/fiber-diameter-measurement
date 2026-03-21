from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox


class MeasurementGroupComboBox(QComboBox):
    """Category combo used inside the measurement table.

    It should only change by explicit click/open interactions, not by wheel
    scrolling while the user is browsing the canvas.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def wheelEvent(self, event) -> None:
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()
