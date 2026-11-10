from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction, QIcon, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QHBoxLayout, QSpinBox, QToolButton, QWidget

from fdm.ui.theme import apply_application_theme
from fdm.ui.widgets import MeasurementToolStrip


@pytest.fixture
def strip(desktop_application):
    widget = MeasurementToolStrip()
    for index in range(8):
        widget.addModeAction(str(index), QAction(f"左侧测量工具 {index}", widget))
    options = QWidget()
    row = QHBoxLayout(options)
    editor = QSpinBox()
    editor.setObjectName("numericOption")
    row.addWidget(editor)
    for index in range(8):
        option = QToolButton()
        option.setText(f"右侧工具选项 {index}")
        row.addWidget(option)
    finish = QToolButton()
    finish.setText("完成")
    cancel = QToolButton()
    cancel.setText("取消")
    widget.setMagicContextWidget(options, pinned=(finish, cancel))
    widget.setMagicContextVisible(True)
    widget.resize(640, widget.sizeHint().height())
    widget.show()
    widget.activateWindow()
    QApplication.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def wheel(widget, *, pixels=QPoint(), angle=QPoint()):
    pos = widget.rect().center()
    QApplication.sendEvent(widget, QWheelEvent(
        QPointF(pos), QPointF(widget.mapToGlobal(pos)), pixels, angle,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    ))


def assert_visible_scroll_icon(button, *, dark):
    # A non-null icon can still be clipped by the application's button padding.
    icon = button.icon()
    painted = button.grab().toImage()
    button.setIcon(QIcon())
    background = button.grab().toImage()
    button.setIcon(icon)
    assert painted.size() == background.size()
    polarity = 1 if dark else -1
    ink = 0
    for y in range(painted.height() // 4, 3 * painted.height() // 4):
        for x in range(painted.width() // 6, 5 * painted.width() // 6):
            delta = painted.pixelColor(x, y).lightness() - background.pixelColor(x, y).lightness()
            if polarity * delta > 20:
                ink += 1
    assert ink >= 12 * painted.devicePixelRatio() ** 2, button.accessibleName()


def test_overflow_arrows_remain_visible_at_both_ends_across_theme_changes(strip, desktop_application):
    original = desktop_application.property("fdmAppliedThemeMode") or "system"
    try:
        for mode in ("light", "dark", "warm", "navy"):
            apply_application_theme(desktop_application, mode)
            desktop_application.processEvents()
            for pane in (strip._primary_scroll, strip._context_scroll):
                bar = pane.area.horizontalScrollBar()
                assert bar.maximum() > bar.minimum()
                for at_start in (True, False):
                    bar.setValue(bar.minimum() if at_start else bar.maximum())
                    assert pane.backButton.isEnabled() == (not at_start)
                    assert pane.forwardButton.isEnabled() == at_start
                    for button in (pane.backButton, pane.forwardButton):
                        assert button.isVisible()
                        assert_visible_scroll_icon(button, dark=mode in ("dark", "navy"))
    finally:
        apply_application_theme(desktop_application, original)


def test_arrows_wheel_and_keyboard_scroll_only_their_own_pane(strip):
    tools, options = strip._primary_scroll, strip._context_scroll
    left = tools.area.horizontalScrollBar()
    right = options.area.horizontalScrollBar()
    assert left.maximum() > 0 and right.maximum() > 0
    tools.forwardButton.click()
    assert left.value() > 0 and right.value() == 0
    tools.backButton.click()
    assert left.value() == 0
    last = strip.buttonForMode("7")
    last.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert left.value() > 0 and last.visibleRegion().contains(last.rect().center())
    previous_left = left.value()
    wheel(options.area.viewport(), pixels=QPoint(-11, 0))
    assert right.value() == 11
    wheel(options.area.viewport(), angle=QPoint(0, -120))
    assert right.value() > 11 and left.value() == previous_left

    # Focus scrolls the editor into view. Its wheel still edits the number.
    editor = strip.findChild(QSpinBox, "numericOption")
    editor.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    previous_right = right.value()
    wheel(editor, angle=QPoint(0, 120))
    assert editor.value() == editor.singleStep()
    assert right.value() == previous_right


def test_splitter_drag_sets_ratio_but_geometry_changes_do_not_emit(strip):
    ratios = []
    strip.splitRatioChanged.connect(ratios.append)
    handle = strip._splitter.handle(1)
    start = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(handle, start - QPoint(80, 0))
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start - QPoint(80, 0))
    QApplication.processEvents()
    left, right = strip._splitter.sizes()
    assert ratios and ratios[-1] == pytest.approx(left / (left + right))
    ratios.clear()
    strip.resize(830, strip.height())
    strip.setMagicContextVisible(False)
    strip.setMagicContextVisible(True)
    strip.setSplitRatio(0.55)
    QApplication.processEvents()
    assert not ratios
    left, right = strip._splitter.sizes()
    assert left / (left + right) == pytest.approx(0.55, abs=0.005)
