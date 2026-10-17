"""Workspace pages contribute geometry only while they are displayed."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize
from PySide6.QtWidgets import QLayout, QSizePolicy, QStackedWidget


class WorkspaceStack(QStackedWidget):
    """Keep retained pages without inheriting every hidden page's minimum.

    QStackedLayout normally computes the largest size hint across all pages.
    That is useful for a small settings dialog, but makes a retained trial or
    native SDK preview enlarge an unrelated workspace after navigation.
    Current-page constraints still apply; scroll areas remain responsible for
    making genuinely large visible content reachable.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.currentChanged.connect(self._current_page_changed)

    def _current_page_changed(self, _index: int) -> None:
        self.layout().invalidate()
        self.updateGeometry()

    def _with_margins(self, size: QSize) -> QSize:
        margins = self.contentsMargins()
        return size.expandedTo(QSize(0, 0)) + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def minimumSizeHint(self) -> QSize:
        page = self.currentWidget()
        if page is None:
            return self._with_margins(QSize(0, 0))
        minimum = page.minimumSizeHint()
        policy = page.sizePolicy()
        # Preserve the page's explicit ability to shrink (for example the
        # welcome illustration). Qt's ordinary layouts honor Ignored too.
        if policy.horizontalPolicy() == QSizePolicy.Policy.Ignored:
            minimum.setWidth(0)
        if policy.verticalPolicy() == QSizePolicy.Policy.Ignored:
            minimum.setHeight(0)
        return self._with_margins(minimum.expandedTo(page.minimumSize()))

    def sizeHint(self) -> QSize:
        page = self.currentWidget()
        if page is None:
            return self.minimumSizeHint()
        return self._with_margins(page.sizeHint()).expandedTo(self.minimumSizeHint())

    def hasHeightForWidth(self) -> bool:
        page = self.currentWidget()
        return page is not None and page.hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:
        page = self.currentWidget()
        if page is None or not page.hasHeightForWidth():
            return -1
        margins = self.contentsMargins()
        height = page.heightForWidth(max(0, width - margins.left() - margins.right()))
        return height + margins.top() + margins.bottom() if height >= 0 else height

    def event(self, event):
        handled = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            # Visible-page contents (status text, scroll areas) can change
            # their hints without changing the selected page.
            self.updateGeometry()
        return handled
