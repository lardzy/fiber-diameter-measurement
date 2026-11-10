"""Deferred UI work must not outlive its Qt owner."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from shiboken6 import isValid

from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import MainWindow


def test_destroy_window_after_records_layout_reset(desktop_application):
    window = MainWindow()
    window.resize(1366, 768)
    window.show()
    desktop_application.processEvents()
    window._reset_workspace_layout()
    assert window._inspector_pages.currentWidget() is window._records_page
    assert window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    desktop_application.processEvents()
    assert not isValid(window)


@pytest.mark.parametrize("destroy_before_dispatch", [False, True])
def test_slide_initial_fit_is_cancelled_when_canvas_is_deleted(
    desktop_application, destroy_before_dispatch,
):
    calls = []

    class TrackedCanvas(DigitalSlideCanvas):
        def _apply_initial_fit(self):
            calls.append(isValid(self))
            if isValid(self):
                super()._apply_initial_fit()

    canvas = TrackedCanvas()
    try:
        canvas.schedule_initial_fit()
        assert not calls
        if destroy_before_dispatch:
            canvas.shutdown()
            canvas.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert not isValid(canvas)
        desktop_application.processEvents()
        assert calls == ([] if destroy_before_dispatch else [True])
    finally:
        if isValid(canvas):
            canvas.shutdown()
            canvas.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
