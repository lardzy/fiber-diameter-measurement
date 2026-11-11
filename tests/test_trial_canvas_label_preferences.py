"""Actual trial labels, font/layout parity, and immutable measurement input."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication

from fdm.live_experiment.frames import FrameHandle
from fdm.trial_preferences import TrialPreferences
from fdm.ui.live_measurement_canvas import LiveMeasurementCanvas
from fdm.ui.trial_label_style import (
    _text_layout, draw_trial_label, trial_label_font, trial_label_size, trial_label_text,
)
from test_live_measurement_candidate_canvas import CanvasController


@pytest.fixture
def trial_canvas(desktop_application):
    controller = CanvasController()
    canvas = LiveMeasurementCanvas(controller)
    canvas.resize(800, 600)
    canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 180, np.uint8),
                                         source_id="image", live=False))
    canvas.show()
    desktop_application.processEvents()
    yield canvas
    canvas.close()
    canvas.deleteLater()


def record(canvas, *, material_id="wool", diameter=18.25, number=7):
    value = canvas.controller.session.add_measurement(
        diameter, material_id, endpoints=((300, 200), (390, 220)),
        provenance={"measurement_number": number})
    canvas.controller.current_records.append(value.id)
    return value


def test_defaults_full_category_and_diameter_value_keep_record_identity(trial_canvas):
    saved = record(trial_canvas)
    session = trial_canvas.controller.session
    item = trial_canvas._overlay_items()[0][0]
    assert item["label"] == session.material("wool").name
    assert "#7" in item["tooltip"] and "18.25 μm" in item["tooltip"]
    session.workspace_options["mode"] = "diameter"
    assert trial_canvas._overlay_items()[0][0]["label"] == "18.25 μm"
    assert saved.provenance["measurement_number"] == 7


def test_color_and_background_use_shared_actual_paint_helper(desktop_application):
    settings = TrialPreferences(color_mode="custom", font_color="#197e27")
    font = trial_label_font(settings)
    size = trial_label_size("绵羊毛", font)
    rect = QRectF(8., 8., size.width(), size.height())

    def paint(preferences, *, semantic=False):
        image = QImage(260, 100, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        draw_trial_label(painter, rect, "绵羊毛", "#ac2100", preferences, semantic_color=semantic)
        painter.end()
        return image

    transparent = paint(settings)
    assert transparent.pixelColor(8, 8).alpha() == 0
    pixels = [transparent.pixelColor(x, y) for y in range(100) for x in range(260)]
    assert any(color.alpha() > 250 and color.name() == "#197e27" for color in pixels)
    background = paint(replace(settings, label_background=True))
    assert background.pixelColor(8, 8) == QColor(255, 255, 255, 232)
    warning = paint(settings, semantic=True)
    assert any(warning.pixelColor(x, y).name() == "#ac2100" and warning.pixelColor(x, y).alpha() > 250
               for y in range(100) for x in range(260))


def test_style_application_changes_layout_but_not_measurements_or_frame(trial_canvas):
    saved = record(trial_canvas)
    session = trial_canvas.controller.session
    before = session.to_dict()
    pixels = trial_canvas.frame.pixels.copy()
    revision = trial_canvas.display_revision
    trial_canvas.repaint()
    original = trial_canvas._label_placements["record:" + saved.id].rect
    settings = TrialPreferences(content_label="number_material", font_size=28,
                                font_bold=True, font_italic=True, label_background=True)
    trial_canvas.set_trial_preferences(settings)
    trial_canvas.repaint()
    item = trial_canvas._overlay_items()[0][0]
    rect = trial_canvas._label_placements[item["key"]].rect
    expected = trial_label_size(item["label"], trial_label_font(settings, trial_canvas.font()))
    assert rect.size() == expected
    assert rect.height() > original.height()
    assert session.to_dict() == before
    assert trial_canvas.display_revision == revision
    assert np.array_equal(trial_canvas.frame.pixels, pixels)
    # Exercise QWidget event dispatch and the real hit path synchronously.
    # QTest.mouseMove can queue a native cursor move on Windows (or target an
    # obscured window); it does not guarantee this widget has received it when
    # it returns. Native activation/hover requires separate desktop validation.
    position = rect.center()
    event = QMouseEvent(QEvent.Type.MouseMove, position, trial_canvas.mapToGlobal(position),
                        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(trial_canvas, event)
    assert "#7" in trial_canvas.toolTip() and trial_canvas._hovered_record_id == saved.id


def test_long_name_wrap_complete_and_font_screen_sized(trial_canvas):
    value = record(trial_canvas)
    session = trial_canvas.controller.session
    materials = session.operators[session.active_operator_id].materials[session.active_direction]
    name = "非常长的完整纤维类别名称和UnbrokenLatinName" * 3
    session.set_materials([replace(item, name=name) if item.id == "wool" else item for item in materials])
    trial_canvas.repaint()
    item = trial_canvas._overlay_items()[0][0]
    assert item["label"] == name
    font = trial_label_font(trial_canvas.trial_preferences, trial_canvas.font())
    full = trial_label_size(name, font)
    single = trial_label_size("绵羊毛", font)
    assert full.width() <= 241 and full.height() > single.height()
    layout, _ = _text_layout(name, font.toString(), 230.)
    assert sum(layout.lineAt(i).textLength() for i in range(layout.lineCount())) == len(name)
    initial_size = trial_canvas._label_placements["record:" + value.id].rect.size()
    # Keep this geometry in view at another scale, and compare screen extents.
    trial_canvas.set_display_mode("percent", 110.)
    trial_canvas.repaint()
    assert trial_canvas._label_placements["record:" + value.id].rect.size() == initial_size


def test_retained_labels_resolve_latest_category_and_color(trial_canvas):
    saved = record(trial_canvas)
    trial_canvas.controller.current_records.clear()
    trial_canvas.controller.selection_retention = SimpleNamespace(visible_marks=lambda: [dict(
        record_id=saved.id, number=7, start=(300, 200), end=(390, 220), color="#abcdef")])
    before = trial_canvas._overlay_items()[0][0]
    trial_canvas.controller.session.edit_measurement(saved.id, material_id="cashmere")
    after = trial_canvas._overlay_items()[0][0]
    assert after["label"] == trial_canvas.controller.session.material("cashmere").name
    assert after["label"] != before["label"] and after["color"] != before["color"]
    assert "#7" in after["tooltip"]


def test_candidates_pending_and_same_frame_layout_stay_stable(trial_canvas):
    record(trial_canvas)
    trial_canvas.controller.candidates = [dict(candidate_id="batch:1", ordinal=4,
        start=(500, 300), end=(530, 300), state="confirming")]
    trial_canvas.controller.pending_classification = [(SimpleNamespace(id="request", measurement_number=9),
        dict(start=(120, 160), end=(150, 180)))]
    trial_canvas.repaint()
    items = trial_canvas._overlay_items()[0]
    assert next(item for item in items if item["kind"] == "candidate")["label"] == "C4 · 待计入"
    assert "#9（未入账）" in next(item for item in items if item["kind"] == "pending")["label"]
    placements = trial_canvas._label_placements
    key = trial_canvas._annotation_cache_key
    trial_canvas.set_frame(FrameHandle.snapshot(np.full((576, 768, 3), 190, np.uint8),
                          source_id="image", sequence=1, live=False))
    trial_canvas.repaint()
    assert trial_canvas._label_placements is placements
    assert trial_canvas._annotation_cache_key == key


@pytest.mark.parametrize("mode,content,expected", [
    ("content", "material", "绵羊毛"), ("content", "number", "#5"),
    ("content", "number_material", "#5 · 绵羊毛"), ("content", "value", "18.25 μm"),
    ("diameter", "value", "18.25 μm"), ("diameter", "number", "#5"),
])
def test_preview_and_canvas_text_contract(mode, content, expected):
    preferences = replace(TrialPreferences(), **{mode + "_label": content})
    assert trial_label_text(preferences, mode=mode, number=5,
                            material_name="绵羊毛", diameter_um=18.25) == expected
