"""Deferred UI work must not outlive its Qt owner."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from shiboken6 import isValid

from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import MainWindow


@pytest.mark.parametrize("destroy_before_dispatch", [False, True])
def test_inspector_restore_is_cancelled_when_window_is_deleted(
    desktop_application, destroy_before_dispatch,
):
    calls = []

    class TrackedWindow(MainWindow):
        def _restore_inspector_section_sizes(self):
            calls.append(isValid(self))
            if isValid(self):
                super()._restore_inspector_section_sizes()

    window = TrackedWindow()
    try:
        # Drain construction-time work, then queue the real settings/layout
        # restore path. Closing before dispatch used to invoke it on dead Qt
        # children and leave an exception pending for the next paint event.
        desktop_application.processEvents()
        calls.clear()
        window._restore_inspector_section_defaults()
        assert not calls
        if destroy_before_dispatch:
            assert window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert not isValid(window)
        desktop_application.processEvents()
        assert calls == ([] if destroy_before_dispatch else [True])
    finally:
        if isValid(window):
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


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
