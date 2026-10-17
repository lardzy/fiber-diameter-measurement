from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPalette, QWheelEvent
from PySide6.QtTest import QTest

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.services.digital_slide_store import DigitalSlideManifest
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas, digital_slide_boundary_color


@pytest.fixture
def canvas(desktop_application):
    widget = DigitalSlideCanvas()
    widget.resize(400, 320)
    document = ImageDocument(
        id="boundary-slide", path="/tmp/boundary.fdmslide",
        image_size=(1000, 600), document_kind="digital_slide",
    )
    image = QImage(200, 100, QImage.Format.Format_RGB32)
    image.fill(QColor("#a9b89c"))
    widget.set_document(document, image)
    widget._slide_manifest = DigitalSlideManifest(
        version=1, width=1000, height=600, viewport_width=200,
        viewport_height=100, focus_levels=[-1, 0, 1],
    )
    widget._browse_center = Point(500, 300)
    widget.fit_native_viewport()
    yield widget
    widget.shutdown()
    widget.clear_document()
    widget.close()
    widget.deleteLater()


@pytest.mark.parametrize(("dx", "dy", "direction"), [
    (-1000, 0, "左边界 ←"), (1000, 0, "右边界 →"),
    (0, -1000, "上边界 ↑"), (0, 1000, "下边界 ↓"),
    (-1000, -1000, "左上边界 ↖"), (1000, -1000, "右上边界 ↗"),
    (-1000, 1000, "左下边界 ↙"), (1000, 1000, "右下边界 ↘"),
])
def test_user_navigation_reports_the_reached_direction(canvas, dx, dy, direction):
    messages = []
    canvas.boundaryFeedbackChanged.connect(messages.append)
    content = canvas._content_rect()
    canvas.move_viewport_by(dx, dy)

    assert messages == [f"已到切片{direction}"]
    assert canvas._content_rect() == content
    assert canvas._boundary_animation.duration() == 800
    canvas._boundary_notice_timer.stop()
    canvas._settle_boundary_feedback()
    assert messages[-1] == direction


def test_held_navigation_does_not_restart_feedback_or_render_work(canvas):
    canvas.move_viewport_by(1000, 0)
    canvas._boundary_animation.setCurrentTime(600)
    canvas._boundary_notice_timer.stop()
    canvas._settle_boundary_feedback()
    before = canvas.browse_view()
    generation = canvas._view_generation
    image_key = canvas._image.cacheKey()
    messages = []
    canvas.boundaryFeedbackChanged.connect(messages.append)
    with (
        patch.object(canvas, "_request_interactive_frames") as render,
        patch.object(canvas, "_publish_viewport_state") as publish,
        patch.object(canvas, "_persist_view_state") as persist,
        patch.object(canvas, "update") as repaint,
    ):
        for _ in range(20):
            canvas.move_viewport_by(12, 0, throttled=True)
    assert canvas.browse_view() == before
    assert canvas._image.cacheKey() == image_key
    assert canvas._view_generation == generation
    assert canvas._boundary_animation.currentTime() == 600
    assert messages == []
    render.assert_not_called()
    publish.assert_not_called()
    persist.assert_not_called()
    repaint.assert_not_called()


def test_feedback_automatically_settles_without_reloading_the_image(canvas):
    canvas.center_on_image_point(Point(1000, 300))
    generation = canvas._view_generation
    with patch.object(canvas, "_request_interactive_frames") as render:
        canvas.move_viewport_by(10, 0)
        assert canvas.boundary_feedback_text() == "已到切片右边界 →"
        QTest.qWait(1700)
        assert canvas.boundary_feedback_text() == "右边界 →"
        assert canvas._boundary_animation.state() == canvas._boundary_animation.State.Stopped
    assert canvas._view_generation == generation
    render.assert_not_called()


def test_tangential_motion_and_leaving_a_corner_keep_the_free_axes_usable(canvas):
    canvas.move_viewport_by(1000, 1000)
    corner = canvas.browse_view().center_px
    canvas._boundary_animation.setCurrentTime(700)
    canvas.move_viewport_by(0, -30)
    assert canvas.browse_view().center_px == Point(corner.x, corner.y - 30)
    assert "右边界" in canvas.boundary_feedback_text()
    assert "下" not in canvas.boundary_feedback_text()
    assert canvas._boundary_animation.currentTime() == 700

    canvas.move_viewport_by(-30, 0)
    assert canvas.boundary_feedback_text() == ""
    assert not canvas._boundary_notice_timer.isActive()
    canvas.move_viewport_by(30, 0)
    assert canvas.boundary_feedback_text() == "已到切片右边界 →"
    assert canvas._boundary_animation.currentTime() == 0


def test_boundary_uses_current_visible_width_instead_of_native_field(canvas):
    canvas.set_view_zoom(canvas.view_zoom() * 2)
    canvas.center_on_image_point(Point(895, 300))
    # The fixed native field already touches X=1000, but the zoomed-in camera
    # still has another 50 source pixels to browse before its own right edge.
    canvas.move_viewport_by(5, 0)
    assert canvas.native_viewport_rect().right() == 1000
    assert canvas.boundary_feedback_text() == ""
    canvas.move_viewport_by(50, 0)
    assert canvas.visible_slide_rect().right() == pytest.approx(1000)
    assert canvas.boundary_feedback_text() == "已到切片右边界 →"


def test_open_fit_locate_and_zoom_are_silent_and_clear_old_feedback(canvas):
    assert canvas.boundary_feedback_text() == ""
    canvas.center_on_image_point(Point(1000, 600))
    assert canvas.boundary_feedback_text() == ""
    canvas.move_viewport_by(5, 0)
    assert canvas.boundary_feedback_text()
    canvas.actual_size()
    assert canvas.boundary_feedback_text() == ""
    canvas.fit_to_view()
    assert canvas.boundary_feedback_text() == ""

    before = canvas.browse_view()
    canvas.move_viewport_by(50, 50)
    assert canvas.browse_view() == before
    assert canvas.boundary_feedback_text() == "已显示整张切片"
    painted = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    painted.fill(Qt.GlobalColor.transparent)
    painter = QPainter(painted)
    canvas._paint_boundary_feedback(painter)
    painter.end()
    assert all(byte == 0 for byte in painted.constBits())


def test_focus_endpoints_and_wheel_never_raise_boundary_feedback(canvas):
    messages = []
    canvas.boundaryFeedbackChanged.connect(messages.append)
    canvas.center_on_image_point(Point(1000, 600))
    canvas.set_focus_index(99)
    wheel = QWheelEvent(
        QPointF(100, 100), QPointF(100, 100), QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    canvas.wheelEvent(wheel)
    assert canvas.focus_index() == 2
    canvas.set_focus_index(-99)
    assert canvas.focus_index() == 0
    assert messages == []
    assert not canvas._boundary_notice_timer.isActive()


@pytest.mark.parametrize("mode", ["step", "smooth", "mouse"])
def test_mouse_and_both_keyboard_modes_use_boundary_feedback(canvas, mode):
    canvas.center_on_image_point(Point(899.9, 300))
    if mode == "mouse":
        canvas._last_mouse_pos = QPointF(100, 100)
        canvas._apply_pan_position(QPointF(80, 100))
    else:
        canvas.set_navigation_mode(mode)
        canvas.keyPressEvent(QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier,
        ))
        canvas.keyReleaseEvent(QKeyEvent(
            QEvent.Type.KeyRelease, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier,
        ))
    assert canvas.boundary_feedback_text() == "已到切片右边界 →"


@pytest.mark.parametrize("dark", [False, True])
def test_feedback_paints_only_in_existing_margin_and_respects_palette(canvas, dark):
    palette = QPalette(canvas.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#202830" if dark else "#ffffff"))
    canvas.setPalette(palette)
    canvas.move_viewport_by(1000, 1000)
    canvas._boundary_animation.stop()
    canvas._set_boundary_opacity(1.0)
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    canvas._paint_boundary_feedback(painter)
    painter.end()

    content = canvas._content_rect().toAlignedRect()
    assert canvas._boundary_paint_region().intersected(content).isEmpty()
    content_image = image.copy(content)
    assert all(byte == 0 for byte in content_image.constBits())
    right = canvas._boundary_edge_rects()["right"].center().toPoint()
    bottom = canvas._boundary_edge_rects()["bottom"].center().toPoint()
    assert image.pixelColor(right) == digital_slide_boundary_color(palette)
    assert image.pixelColor(bottom) == digital_slide_boundary_color(palette)
    assert image.pixelColor(canvas._boundary_edge_rects()["left"].center().toPoint()).alpha() == 0


@pytest.mark.parametrize("operation", ["hide", "clear", "shutdown", "resize"])
def test_feedback_lifecycle_does_not_leak_to_another_view(canvas, operation):
    canvas.move_viewport_by(1000, 0)
    if operation == "hide":
        from PySide6.QtGui import QHideEvent
        canvas.hideEvent(QHideEvent())
    elif operation == "resize":
        from PySide6.QtGui import QResizeEvent
        from PySide6.QtCore import QSize
        canvas.resizeEvent(QResizeEvent(QSize(500, 400), canvas.size()))
    elif operation == "clear":
        canvas.clear_document()
    else:
        canvas.shutdown()
    assert canvas.boundary_feedback_text() == ""
    assert not canvas._boundary_notice_timer.isActive()
    assert canvas._boundary_animation.state() == canvas._boundary_animation.State.Stopped
