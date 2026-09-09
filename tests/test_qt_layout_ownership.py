"""Qt must invalidate the old layout item before a widget changes layouts."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)
from shiboken6 import createdByPython, getAllValidWrappers, isValid

from fdm.ui.layout_utils import detach_widget_from_layout
from fdm.ui.main_window import MainWindow
from fdm.ui.statistics_distribution import StatisticsDistributionWidget
from fdm.ui.widgets import CollapsibleSection, FlowLayout, MeasurementToolStrip


def _item_for_widget(layout: QLayout, widget):
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is widget:
            return item
        child = item.layout()
        if child is not None:
            found = _item_for_widget(child, widget)
            if found is not None:
                return found
    return None


def test_unmanaged_widget_lookup_does_not_wrap_native_main_window_items():
    window = QMainWindow()
    window.setCentralWidget(QWidget())
    window.addToolBar("Tools")
    widget = QWidget(window)
    try:
        before = {
            id(item) for item in getAllValidWrappers()
            if isinstance(item, QWidgetItem) and not createdByPython(item)
        }
        assert not detach_widget_from_layout(widget)
        after = {
            id(item) for item in getAllValidWrappers()
            if isinstance(item, QWidgetItem) and not createdByPython(item)
        }
        # Native main-window items can be deleted by restoreState/fullscreen
        # without Shiboken notification. Do not materialize unrelated wrappers
        # while looking for a newly created (not yet laid out) widget.
        assert after == before
        assert widget.parentWidget() is window
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_capture_task_bar_invalidates_every_old_layout_item():
    class TrackedWindow(MainWindow):
        def _install_capture_task_bar(self):
            self.moved_controls = (
                self._digital_slide_readiness_label,
                self._digital_slide_plan_summary_label,
                self._digital_slide_progress_bar,
                self._digital_slide_start_button,
                self._digital_slide_stop_button,
            )
            self.old_items = tuple(
                _item_for_widget(widget.parentWidget().layout(), widget)
                for widget in self.moved_controls
            )
            assert all(item is not None for item in self.old_items)
            super()._install_capture_task_bar()

    window = TrackedWindow()
    try:
        assert all(not isValid(item) for item in window.old_items)
        for widget in window.moved_controls:
            assert isValid(widget)
            assert widget.parentWidget() is window._capture_task_bar
            assert window._capture_task_bar.layout().indexOf(widget) >= 0
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("source_flow", [False, True])
def test_flow_layout_transfer_releases_old_item_without_deleting_widget(source_flow):
    source = QWidget()
    destination = QWidget()
    source_layout = FlowLayout(source) if source_flow else QVBoxLayout(source)
    destination_layout = FlowLayout(destination)
    widget = QLabel("Transfer", source)
    source_layout.addWidget(widget)
    old_item = source_layout.itemAt(0)
    try:
        destination_layout.addWidget(widget)
        assert source_layout.count() == 0
        assert not isValid(old_item)
        assert isValid(widget)
        assert widget.parentWidget() is destination
        assert destination_layout.count() == 1
        # Reusing the same widget must not leave two items pointing at it.
        destination_layout.addWidget(widget)
        assert destination_layout.count() == 1
    finally:
        source.deleteLater()
        destination.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_collapsible_section_transfer_and_replacement_release_layout_items():
    source = QWidget()
    source_layout = QVBoxLayout(source)
    section = CollapsibleSection("Content")
    widget = QLabel("Original", source)
    source_layout.addWidget(widget)
    source_item = source_layout.itemAt(0)
    replacement = QLabel("Replacement")
    try:
        section.setContentWidget(widget)
        assert not isValid(source_item)
        section_item = section.contentLayout.itemAt(0)
        section.setContentWidget(replacement)
        assert not isValid(section_item)
        assert isValid(widget)
        assert widget.parentWidget() is None
        section.setContentWidget(replacement)
        assert section.contentLayout.count() == 1
        assert section.contentLayout.itemAt(0).widget() is replacement
    finally:
        source.deleteLater()
        widget.deleteLater()
        section.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("kind", ["Magic", "Count", "Preview", "Path", "Construction"])
def test_tool_context_transfer_from_nested_layout_invalidates_old_item(kind):
    source = QWidget()
    source_layout = QVBoxLayout(source)
    row = QHBoxLayout()
    source_layout.addLayout(row)
    widget = QLabel("Context", source)
    row.addWidget(widget)
    old_item = row.itemAt(0)
    strip = MeasurementToolStrip()
    try:
        setter = getattr(strip, f"set{kind}ContextWidget")
        setter(widget)
        assert row.count() == 0
        assert not isValid(old_item)
        assert isValid(widget)
        assert widget.parentWidget() is strip._context_host
        setter(widget)
        assert strip._context_layout.count() == 1
    finally:
        source.deleteLater()
        strip.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_statistics_reflow_releases_replaced_layout_items():
    panel = StatisticsDistributionWidget()
    try:
        panel._apply_responsive_layout(480)
        for width in (800, 1280, 480):
            old_items = [
                layout.itemAt(index)
                for layout in (panel._cards_layout, panel._controls_layout)
                for index in range(layout.count())
            ]
            panel._apply_responsive_layout(width)
            assert all(not isValid(item) for item in old_items)
            assert panel._cards_layout.count() == len(panel._cards)
            assert panel._controls_layout.count() == len(panel._control_widgets)
            assert all(isValid(widget) for widget in panel._cards + panel._control_widgets)
    finally:
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
