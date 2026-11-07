"""Alt/Option starts a separate manual measurement at the exact endpoint under the pointer."""

from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.geometry import Line, Point
from fdm.models import Measurement
from test_metrology_workspace import mounted_window, professional_slide


def send_mouse(canvas, kind, pos, modifiers=Qt.KeyboardModifier.NoModifier):
    left = Qt.MouseButton.LeftButton
    button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else left
    pressed = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else left
    QApplication.sendEvent(canvas, QMouseEvent(kind, pos, canvas.mapToGlobal(pos), button, pressed, modifiers))


def drag(canvas, start, end, *, extend=False):
    modifiers = Qt.KeyboardModifier.AltModifier if extend else Qt.KeyboardModifier.NoModifier
    if extend:
        QTest.keyPress(canvas, Qt.Key.Key_Alt)
    send_mouse(canvas, QEvent.Type.MouseButtonPress, start, modifiers)
    if extend:
        QTest.keyRelease(canvas, Qt.Key.Key_Alt)
    send_mouse(canvas, QEvent.Type.MouseMove, end)
    send_mouse(canvas, QEvent.Type.MouseButtonRelease, end)


def assert_point(actual, expected):
    assert (actual.x, actual.y) == pytest.approx((expected.x, expected.y), abs=1e-9, rel=0)


@pytest.mark.parametrize("source_selected", [True, False], ids=["selected", "unselected"])
@pytest.mark.parametrize("snap_enabled", [False, True], ids=["snap-off", "snap-on"])
def test_chain_from_snapped_endpoint_creates_independent_records_and_undo(mounted_window, source_selected, snap_enabled):
    window, document, _, _ = mounted_window()
    canvas = window.current_canvas()
    window.set_tool_mode("manual")
    canvas.set_settings(replace(window._app_settings, object_snap_enabled=snap_enabled))
    a, b = Point(22.5, 55.75), Point(83.25, 60.375)
    source = Measurement("original", document.id, None, "snap",
                         line_px=Line(Point(18, 50), Point(90, 68)), snapped_line_px=Line(a, b))
    window._append_new_measurement(document, source, label="已有直径")
    if not source_selected:
        window._on_canvas_line_committed(document.id, "manual", Line(Point(52, 138), Point(90, 150)))
    assert (document.view_state.selected_measurement_id == source.id) == source_selected
    originals = [m.to_dict() for m in document.measurements]

    c, d = Point(135, 85), Point(204, 88)
    drag(canvas, canvas.image_to_widget(b) + QPointF(1.25, -1.5), canvas.image_to_widget(c), extend=True)
    segment_bc = document.measurements[-1]
    assert len(document.measurements) == len(originals) + 1
    assert segment_bc.mode == "manual"
    assert segment_bc.effective_line().start == b
    assert_point(segment_bc.effective_line().end, c)

    drag(canvas, canvas.image_to_widget(c) + QPointF(-1, 1), canvas.image_to_widget(d), extend=True)
    segment_cd = document.measurements[-1]
    assert len(document.measurements) == len(originals) + 2
    assert segment_cd.id != segment_bc.id
    assert segment_cd.effective_line().start == segment_bc.effective_line().end
    assert_point(segment_cd.effective_line().end, d)
    assert [m.to_dict() for m in document.measurements[:len(originals)]] == originals

    window.undo_current_document()
    assert document.get_measurement(segment_cd.id) is None
    assert document.get_measurement(segment_bc.id).effective_line().start == b
    window.undo_current_document()
    assert [m.to_dict() for m in document.measurements] == originals


def test_manual_alt_cancellation_normal_adjustment_and_empty_space(mounted_window):
    window, document, _, _ = mounted_window()
    canvas = window.current_canvas()
    window.set_tool_mode("manual")
    canvas.set_settings(replace(window._app_settings, object_snap_enabled=False))
    window._on_canvas_line_committed(document.id, "manual", Line(Point(45, 45), Point(91, 62)))
    source = document.measurements[0]
    before = source.to_dict()
    end = canvas.image_to_widget(source.effective_line().end)
    near = end + QPointF(1.5, 1.0)
    destination = canvas.image_to_widget(Point(136, 84))
    for cancel in (False, True):
        send_mouse(canvas, QEvent.Type.MouseButtonPress, near, Qt.KeyboardModifier.AltModifier)
        if cancel:
            send_mouse(canvas, QEvent.Type.MouseMove, destination)
            QTest.keyClick(canvas, Qt.Key.Key_Escape)
        send_mouse(canvas, QEvent.Type.MouseButtonRelease, destination if cancel else near)
        assert [m.to_dict() for m in document.measurements] == [before]

    drag(canvas, end, destination)
    assert len(document.measurements) == 1
    edited = document.get_measurement(source.id).effective_line()
    assert_point(edited.start, Point(45, 45))
    assert_point(edited.end, Point(136, 84))
    away = Point(25, 130)
    drag(canvas, canvas.image_to_widget(away), canvas.image_to_widget(Point(72, 154)), extend=True)
    assert len(document.measurements) == 2
    assert_point(document.measurements[-1].effective_line().start, away)
    assert document.get_measurement(source.id).effective_line() == edited


def test_alt_in_edge_snap_tool_still_requires_the_second_click(mounted_window):
    window, document, _, _ = mounted_window()
    window._on_canvas_line_committed(document.id, "manual", Line(Point(30, 50), Point(90, 72)))
    window.set_tool_mode("snap")
    canvas = window.current_canvas()
    canvas.set_settings(replace(window._app_settings, object_snap_enabled=False))
    start, end = Point(90, 72), Point(140, 80)
    drag(canvas, canvas.image_to_widget(start), canvas.image_to_widget(end), extend=True)
    assert len(document.measurements) == 1
    assert document.measurements[0].effective_line().end == start
    position = canvas.image_to_widget(end)
    send_mouse(canvas, QEvent.Type.MouseButtonPress, position)
    send_mouse(canvas, QEvent.Type.MouseButtonRelease, position)
    assert [measurement.mode for measurement in document.measurements] == ["manual", "snap"]


def test_alt_drag_on_slide_uses_global_measurement_coordinates(professional_slide):
    window, document, _, _, _ = professional_slide
    canvas = window.current_canvas()
    assert canvas.viewport_origin() != Point(0, 0)
    window.set_tool_mode("manual")
    canvas.set_settings(replace(window._app_settings, object_snap_enabled=False))
    a, b, c = Point(180, 160), Point(229.125, 171.25), Point(278, 198)
    window._on_canvas_line_committed(document.id, "manual", Line(a, b))
    drag(canvas, canvas.image_to_widget(b) + QPointF(1.0, -1.0), canvas.image_to_widget(c), extend=True)
    assert len(document.measurements) == 2
    assert document.measurements[0].effective_line() == Line(a, b)
    assert document.measurements[1].effective_line().start == b
    assert_point(document.measurements[1].effective_line().end, c)


def test_extension_keeps_fractional_endpoint_in_last_image_pixel(mounted_window):
    window, document, _, _ = mounted_window()
    window.set_tool_mode("manual")
    canvas = window.current_canvas()
    canvas.set_settings(replace(window._app_settings, object_snap_enabled=False))
    width, height = canvas._image_size()
    endpoint = Point(width - 0.5, height / 2 + 0.25)
    window._on_canvas_line_committed(document.id, "manual", Line(Point(64, 44), endpoint))
    destination = Point(width - 38, height / 2 + 28)
    drag(canvas, canvas.image_to_widget(endpoint), canvas.image_to_widget(destination), extend=True)
    assert len(document.measurements) == 2
    assert document.measurements[1].effective_line().start == endpoint
    assert_point(document.measurements[1].effective_line().end, destination)


@pytest.mark.parametrize(("modifier", "expected_end"), [
    (Qt.KeyboardModifier.ControlModifier, Point(140.5, 76.5)),
    (Qt.KeyboardModifier.ShiftModifier, Point(140.2, 61.75)),
])
def test_nearby_endpoints_use_the_nearest_and_only_constrain_the_free_end(mounted_window, modifier, expected_end):
    window, document, _, _ = mounted_window()
    canvas = window.current_canvas()
    window.set_tool_mode("manual")
    canvas.set_settings(replace(window._app_settings, object_snap_enabled=False))
    anchor = Point(81.125, 61.75)
    window._on_canvas_line_committed(document.id, "manual", Line(anchor, Point(40, 125)))
    nearby = canvas.widget_to_image(canvas.image_to_widget(anchor) + QPointF(3.0, 0))
    window._on_canvas_line_committed(document.id, "manual", Line(Point(53, 33), nearby))
    press = canvas.image_to_widget(anchor) + QPointF(0.5, 0)
    send_mouse(canvas, QEvent.Type.MouseButtonPress, press, Qt.KeyboardModifier.AltModifier | modifier)
    # Some input devices only report the final movement with the release.
    send_mouse(canvas, QEvent.Type.MouseButtonRelease, canvas.image_to_widget(Point(140.2, 76.4)), modifier)
    assert len(document.measurements) == 3
    assert document.measurements[-1].effective_line().start == anchor
    assert_point(document.measurements[-1].effective_line().end, expected_end)
