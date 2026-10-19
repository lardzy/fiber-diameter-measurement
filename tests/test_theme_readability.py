from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPalette, QWheelEvent
from PySide6.QtWidgets import QDialogButtonBox

from fdm.models import ImageDocument
from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
from fdm.services.export_service import ExportSelection
from fdm.settings import AppSettings
from fdm.ui import theme
from fdm.ui.dialogs import ExportOptionsDialog, SettingsDialog
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas


MODES = ("light", "warm", "dark", "navy")


def contrast(first: QColor, second: QColor) -> float:
    def luminance(color):
        channels = (color.redF(), color.greenF(), color.blueF())
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in channels]
        return sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))
    low, high = sorted((luminance(first), luminance(second)))
    return (high + .05) / (low + .05)


@pytest.fixture
def themed_app(desktop_application):
    app = desktop_application
    original = app.property("fdmAppliedThemeMode") or "dark"
    yield app
    theme.apply_application_theme(app, original)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.mark.parametrize("underlay", ("#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"))
def test_image_notice_text_remains_readable_over_extreme_image_colors(underlay):
    background, foreground = theme.canvas_notice_colors()
    image_color = QColor(underlay)
    alpha = background.alphaF()
    composite = QColor(*(round(a * alpha + b * (1 - alpha)) for a, b in
                         zip(background.getRgb()[:3], image_color.getRgb()[:3])))
    assert contrast(foreground, composite) >= 4.5


def test_candidate_warnings_follow_theme_without_recoloring_measurement_data(themed_app):
    from fdm.live_experiment.models import MaterialSnapshot
    from fdm.ui.live_experiment_workspace import CandidatesModel
    model = CandidatesModel()
    model.set_candidates([{"width_px": 20}], 1, [MaterialSnapshot("test", "测试", 1.3, upper_limit=10)])
    for mode in MODES:
        theme.apply_application_theme(themed_app, mode)
        palette = themed_app.palette()
        for assigned in (None, "test"):
            model.assignments[0] = assigned
            color = model.data(model.index(0, 4), Qt.ItemDataRole.ForegroundRole)
            for role in (QPalette.ColorRole.Base, QPalette.ColorRole.AlternateBase):
                assert contrast(color, palette.color(role)) >= 4.5


def test_open_settings_and_export_dialogs_follow_explicit_and_system_theme_changes(themed_app):
    app = themed_app
    settings = SettingsDialog(AppSettings(), document=None)
    export = ExportOptionsDialog(ExportSelection.all_enabled(), allow_all_scope=True)
    widgets = ((settings, settings._settings_navigation, settings._settings_page_description),
               (export, export._export_navigation, export._export_page_description))
    try:
        for dialog, navigation, _ in widgets:
            dialog.show()
            navigation.setCurrentRow(0)
        for mode, resolved in [(mode, mode) for mode in MODES] + [("system", "light"), ("system", "dark")]:
            with patch.object(theme, "_system_color_mode", return_value=resolved):
                theme.apply_application_theme(app, mode)
                app.processEvents()
            expected = theme.build_theme_palette(resolved)
            for dialog, navigation, description in widgets:
                image = navigation.viewport().grab().toImage()
                rect = navigation.visualItemRect(navigation.currentItem())
                background = image.pixelColor(rect.right() - 12, rect.center().y())
                assert background == expected.color(QPalette.ColorRole.Highlight)
                selected_text = expected.color(QPalette.ColorRole.HighlightedText).getRgb()[:3]
                assert sum(max(abs(a - b) for a, b in zip(image.pixelColor(x, y).getRgb()[:3], selected_text)) < 35
                           for y in range(rect.top() + 4, rect.bottom() - 4)
                           for x in range(rect.left() + 9, rect.right() - 12)) > 3
                text_color = description.palette().color(QPalette.ColorRole.WindowText)
                assert text_color == QColor(theme.semantic_colors(expected).muted)
                assert contrast(text_color, dialog.palette().color(QPalette.ColorRole.Window)) >= 4.5
            button = export._button_box.button(QDialogButtonBox.StandardButton.Ok)
            assert button.palette().color(QPalette.ColorRole.ButtonText) == expected.color(QPalette.ColorRole.HighlightedText)
            assert contrast(button.palette().color(QPalette.ColorRole.ButtonText),
                            button.palette().color(QPalette.ColorRole.Button)) >= 4.5
    finally:
        for dialog, _, _ in widgets:
            dialog.close()
            dialog.deleteLater()


def test_actual_large_area_browse_notice_stays_light_after_hot_theme_switch(themed_app, tmp_path, monkeypatch):
    from fdm.ui import digital_slide_canvas

    app = themed_app
    monkeypatch.setattr(digital_slide_canvas, "digital_slide_render_cache_directory", lambda: tmp_path / "cache")
    source = tmp_path / "contrast.fdmslide"
    manifest = DigitalSlideManifest(1, 400, 320, 100, 80, [0])
    store = DigitalSlideStore.create(source, manifest)
    tile = QImage(400, 320, QImage.Format.Format_RGB32)
    tile.fill(QColor("#8F7F6F"))
    store.write_tile(DigitalSlideTile(0, 0, 0, 400, 320), tile)
    canvas = DigitalSlideCanvas()
    canvas.resize(1000, 700)
    canvas.set_settings(AppSettings(digital_slide_render_cache_gib=0))
    canvas.set_slide_document(ImageDocument(id="readable", path=str(source), image_size=(400, 320),
                                            document_kind="digital_slide"), store)
    canvas.show()
    try:
        end = time.monotonic() + 5
        while not canvas.pixel_work_enabled() and time.monotonic() < end:
            app.processEvents()
            time.sleep(.01)
        assert canvas.pixel_work_enabled()
        # Exercise the real wheel handler instead of assigning browse flags.
        center = QPointF(canvas.rect().center())
        event = QWheelEvent(center, center, QPoint(), QPoint(0, -2400), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
        app.sendEvent(canvas, event)
        assert canvas.large_area_browse_active()
        for mode in MODES:
            theme.apply_application_theme(app, mode)
            app.processEvents()
            image = canvas.grab().toImage()
            bottom = round(canvas._content_rect().bottom())
            # The uniform brown image cannot provide these light pixels: they
            # must belong to the actual visible notice text, not a palette mock.
            light_pixels = sum(image.pixelColor(x, y).lightnessF() > .85
                               for y in range(bottom - 35, bottom - 8)
                               for x in range(canvas.width() // 2 - 170, canvas.width() // 2 + 170))
            assert light_pixels > 20, (mode, light_pixels)
        assert list(store.iter_tiles())[0][1] == tile
    finally:
        canvas.shutdown()
        canvas.clear_document()
        canvas.close()
        canvas.deleteLater()
        store.close()
