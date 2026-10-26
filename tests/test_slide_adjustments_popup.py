from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPalette, QWheelEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import isValid

from fdm.ui.icons import themed_icon
from fdm.ui.slide_brightness_control import SlideBrightnessButton, SlideBrightnessPopup


class Canvas(QWidget):
    displayAdjustmentsChanged = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.document_id = "first"
        self.display_brightness = 0
        self.display_exposure_tenths = 0

    @property
    def display_adjustments(self):
        return self.display_brightness, self.display_exposure_tenths

    def set_display_adjustments(self, brightness, exposure_tenths):
        self.display_brightness, self.display_exposure_tenths = brightness, exposure_tenths
        self.displayAdjustmentsChanged.emit(brightness, exposure_tenths)


@pytest.fixture
def controls(desktop_application):
    canvas = Canvas()
    popup = SlideBrightnessPopup(canvas)
    popup.show()
    desktop_application.processEvents()
    yield canvas, popup
    if isValid(popup):
        popup.close()
        popup.deleteLater()
    if isValid(canvas):
        canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_pair_previews_coalesce_without_committing_then_release_once(controls):
    canvas, popup = controls
    previews = QSignalSpy(canvas.displayAdjustmentsChanged)
    commits = QSignalSpy(popup.adjustmentsCommitted)
    popup.slider.setValue(12)
    popup.exposure_spin_box.setValue(1.3)
    popup.slider.setValue(45)
    assert canvas.display_adjustments == (0, 0)
    assert popup.exposure_slider.value() == 13
    popup.flush()
    assert canvas.display_adjustments == (45, 13)
    assert previews.count() == 1 and commits.count() == 0
    popup.exposure_slider.sliderReleased.emit()
    assert commits.count() == 1 and commits.at(0) == [45, 13]
    popup.spin_box.editingFinished.emit()
    popup.close()
    assert commits.count() == 1


def test_timer_flush_never_persists_and_keeps_processing_adjusted_cadence(controls):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    now = [10_000_000_000]
    apply = canvas.set_display_adjustments

    def slow_apply(brightness, exposure_tenths):
        apply(brightness, exposure_tenths)
        now[0] += 15_000_000

    with (
        patch("fdm.ui.slide_brightness_control.monotonic_ns", side_effect=lambda: now[0]),
        patch.object(canvas, "set_display_adjustments", side_effect=slow_apply),
    ):
        popup.slider.setValue(20)
        assert popup._timer.interval() == 0
        popup.flush()
        popup.exposure_slider.setValue(-5)
        assert popup._timer.interval() == 18
        now[0] += 4_000_000
        popup.slider.setValue(25)
        assert popup._timer.interval() == 18  # Same batch does not restart.
        popup.flush()
    assert canvas.display_adjustments == (25, -5)
    assert commits.count() == 0
    popup.commit()
    assert commits.count() == 1


def test_remember_toggle_flushes_whole_pair_without_duplicate_commit(controls):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    toggles = []
    popup.rememberToggled.connect(lambda checked: toggles.append((checked, canvas.display_adjustments)))
    popup.exposure_spin_box.setValue(-1.6)
    popup.slider.setValue(31)
    popup.remember_checkbox.click()
    assert toggles == [(True, (31, -16))]
    assert commits.count() == 0
    popup.close()
    assert commits.count() == 0


def test_open_sync_and_close_without_edits_do_not_commit(controls):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    toggles = QSignalSpy(popup.rememberToggled)
    canvas.set_display_adjustments(-40, 18)
    popup.sync_adjustments(-40, 18)
    popup.sync_remember(True)
    assert popup.slider.value() == -40
    assert popup.exposure_spin_box.value() == 1.8
    popup.close()
    assert commits.count() == 0 and toggles.count() == 0


def test_reset_commits_neutral_pair_even_when_canvas_already_neutral(controls):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    popup.sync_remember(True)
    popup.reset_button.click()
    assert commits.count() == 1 and commits.at(0) == [0, 0]
    popup.slider.setValue(-70)
    popup.exposure_slider.setValue(26)
    popup.reset_button.click()
    assert canvas.display_adjustments == (0, 0)
    assert popup.slider.value() == 0 and popup.exposure_spin_box.value() == 0
    assert popup.remember_checkbox.isChecked()
    assert commits.count() == 2 and commits.at(1) == [0, 0]
    popup.close()
    assert commits.count() == 2


def test_decimal_input_and_limits_are_exact_tenths(controls):
    canvas, popup = controls
    assert popup.exposure_spin_box.singleStep() == .1
    for value, ticks in [(0.1, 1), (-0.1, -1), (2.9, 29), (-3.9, -39), (9.0, 40), (-9.0, -40)]:
        popup.exposure_spin_box.setValue(value)
        popup.exposure_spin_box.editingFinished.emit()
        assert canvas.display_exposure_tenths == ticks
        assert popup.exposure_slider.value() == ticks
    popup.exposure_slider.setValue(7)
    assert popup.exposure_spin_box.value() == .7


@pytest.mark.parametrize("text,ticks", [("1.3", 13), ("-2.7", -27), ("0.1", 1)])
def test_exposure_keyboard_input_keeps_decimal_text_until_finished(controls, text, ticks):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    popup.exposure_spin_box.setFocus()
    popup.exposure_spin_box.selectAll()
    QTest.keyClicks(popup.exposure_spin_box, text)
    assert commits.count() == 0
    QTest.keyClick(popup.exposure_spin_box, Qt.Key.Key_Return)
    assert popup.exposure_spin_box.value() == ticks / 10
    assert popup.exposure_slider.value() == ticks
    assert canvas.display_adjustments == (0, ticks)
    assert commits.count() == 1 and commits.at(0) == [0, ticks]


@pytest.mark.parametrize("text,value", [("75", 75), ("-65", -65), ("100", 100)])
def test_brightness_keyboard_input_keeps_all_digits_until_finished(controls, text, value):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    popup.spin_box.setFocus()
    popup.spin_box.selectAll()
    QTest.keyClicks(popup.spin_box, text)
    assert commits.count() == 0
    QTest.keyClick(popup.spin_box, Qt.Key.Key_Return)
    assert popup.spin_box.value() == value and popup.slider.value() == value
    assert canvas.display_adjustments == (value, 0)
    assert commits.count() == 1 and commits.at(0) == [value, 0]


@pytest.mark.parametrize("invalidate", ["rebind", "delete"])
def test_invalid_canvas_drops_preview_commit_and_remember_request(controls, invalidate):
    canvas, popup = controls
    commits = QSignalSpy(popup.adjustmentsCommitted)
    toggles = QSignalSpy(popup.rememberToggled)
    popup.slider.setValue(60)
    popup.exposure_slider.setValue(20)
    if invalidate == "rebind":
        canvas.document_id = "replacement"
    else:
        canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    popup.commit()
    popup.remember_checkbox.click()
    assert popup.canvas is None
    assert commits.count() == 0 and toggles.count() == 0
    if invalidate == "rebind":
        assert canvas.display_adjustments == (0, 0)


def test_all_adjustment_controls_consume_wheel(controls):
    canvas, popup = controls
    for control in (popup, popup.slider, popup.spin_box, popup.exposure_slider, popup.exposure_spin_box):
        event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(control, event)
        assert event.isAccepted()
    assert popup.spin_box.value() == 0 and popup.exposure_spin_box.value() == 0
    assert canvas.display_adjustments == (0, 0)


def test_persistence_error_is_visible_and_clearable(controls):
    _, popup = controls
    popup.set_persistence_error("设置保存失败，本次查看仍然有效。")
    assert popup.persistence_error_label.isVisible()
    assert "保存失败" in popup.persistence_error_label.text()
    popup.set_persistence_error("")
    assert not popup.persistence_error_label.isVisible()


def test_button_uses_constant_text_icon_geometry_and_theme_palette(desktop_application):
    button = SlideBrightnessButton()
    try:
        button.show()
        desktop_application.processEvents()
        assert button.text() == "亮度/曝光"
        assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        assert not button.icon().isNull()
        before = button.sizeHint()
        for text in ("亮度：0；曝光：0.0 EV", "亮度：−100；曝光：−4.0 EV", "亮度：100；曝光：4.0 EV"):
            button.setToolTip(text)
            assert button.sizeHint() == before
        palette = QPalette(button.palette())
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ef3020"))
        button.setPalette(palette)
        desktop_application.processEvents()
        assert button.sizeHint() == before
        image = button.icon().pixmap(QSize(36, 36)).toImage()
        opaque = [image.pixelColor(x, y) for y in range(image.height()) for x in range(image.width())
                  if image.pixelColor(x, y).alpha() > 240]
        assert opaque and all(color.red() > color.green() * 2 for color in opaque)
        font = QFont(button.font())
        font.setPointSize(font.pointSize() + 6)
        button.setFont(font)
        assert button.sizeHint().width() > before.width()
    finally:
        button.close()
        button.deleteLater()


@pytest.mark.parametrize("dpr", [1.0, 1.5, 2.0])
def test_sun_icon_renders_at_requested_device_scale(desktop_application, dpr):
    icon = themed_icon("display_adjustments", color="#202020")
    pixmap = icon.pixmap(QSize(20, 20), dpr, QIcon.Mode.Normal, QIcon.State.Off)
    assert not pixmap.isNull()
    assert pixmap.devicePixelRatio() == dpr
    assert pixmap.size() == QSize(round(20 * dpr), round(20 * dpr))
    image = pixmap.toImage()
    assert any(image.pixelColor(x, y).alpha() for x in range(image.width()) for y in range(image.height()))
