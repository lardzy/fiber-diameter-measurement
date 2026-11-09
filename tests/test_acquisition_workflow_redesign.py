"""Regression routes use MainWindow actions and a deterministic fake camera."""
from pathlib import Path
import importlib.util
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

_path = Path(__file__).resolve().parents[1] / "scripts/validation/validate_workflow_redesign.py"
_spec = importlib.util.spec_from_file_location("workflow_redesign_validation", _path)
workflow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workflow)


@pytest.fixture
def harness(tmp_path):
    with workflow.WorkflowHarness(tmp_path / "profile") as instance:
        yield instance


@pytest.mark.parametrize("size", [(1366, 768), (1280, 720), (1366, 690), (1280, 670)])
def test_two_hundred_page_switches_twenty_roundtrips_preserve_shell_and_clean_draft(harness, size):
    harness.size = size
    harness.window.resize(*size)
    workflow.settle()
    result = workflow.validate_transitions(harness)
    assert result["backend_starts"] == 21
    assert not result["unexpected_windows"]
    assert not result["project_dirty"]


def test_four_display_modes_restore_across_all_pages_stop_and_restart(harness):
    assert len(workflow.validate_display_modes(harness)) == 4


def test_layout_only_roundtrips_do_not_move_preview_center(harness):
    harness.enter()
    binding = harness.window._preview_display_binding
    source = binding._source
    expected = binding.store.center(*source)
    for _ in range(20):
        for index in (1, 2, 0):
            harness.select(index)
        harness.leave()
        harness.enter()
        canvas = harness.window._preview_canvas
        point = canvas.widget_to_image(QPointF(canvas.width() / 2, canvas.height() / 2))
        actual = (point.x / source[1][0], point.y / source[1][1])
        # Scrollbars restore to integer logical pixels: at most one source
        # pixel of rounding, never accumulating across navigation cycles.
        assert actual == pytest.approx(expected, abs=1 / min(source[1]))
        assert binding.store.center(*source) == pytest.approx(expected, abs=1 / min(source[1]))
    harness.assert_clean()


def test_real_preview_pan_survives_pages_resize_stop_and_new_window(tmp_path):
    profile = tmp_path / "pan-profile"
    with workflow.WorkflowHarness(profile) as harness:
        harness.enter()
        canvas = harness.window._preview_canvas
        binding = harness.window._preview_display_binding
        binding.set_mode("percent", 200)
        workflow.settle()
        origin = canvas.rect().center()
        QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=origin)
        QTest.mouseMove(canvas, origin + QPoint(100, 70))
        QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=origin + QPoint(100, 70))
        workflow.settle()
        binding.flush()
        source = binding._source
        expected = (.5 - 100 / 2 / source[1][0], .5 - 70 / 2 / source[1][1])
        assert binding.store.center(*source) == pytest.approx(expected, abs=.001)
        for _ in range(3):
            for index in (1, 2, 0):
                harness.select(index)
            for size in ((1920, 1080), (1366, 768)):
                harness.size = size
                harness.window.resize(*size)
                workflow.settle()
            harness.leave()
            harness.enter()
            assert binding.store.center(*source) == pytest.approx(expected, abs=.001)
        harness.assert_clean()
    with workflow.WorkflowHarness(profile) as rebuilt:
        rebuilt.enter()
        binding = rebuilt.window._preview_display_binding
        canvas = rebuilt.window._preview_canvas
        point = canvas.widget_to_image(QPointF(canvas.width() / 2, canvas.height() / 2))
        assert canvas.view_zoom() == 2
        assert (point.x / source[1][0], point.y / source[1][1]) == pytest.approx(expected, abs=.001)
        assert binding.store.center(*source) == pytest.approx(expected, abs=.001)
        rebuilt.assert_clean()


def test_calibrated_magnification_has_same_physical_scale_on_all_three_pages(harness):
    from fdm.ui.live_display import save_screen_ruler
    harness.enter()
    save_screen_ruler(harness.window.screen(), 125)  # 500 screen pixels span 125 mm.
    page = harness.select(1)
    page.ruler_choice.setCurrentIndex(1)
    page.ruler_choice.activated.emit(1)  # 2 image pixels / μm.
    page.display_binding.set_mode("magnification", 200)
    workflow.settle()
    for index in (2, 0, 1):
        page = harness.select(index)
        if index:
            assert page.canvas.display_mode == "magnification"
            actual = page.canvas._transform()[0] * page.canvas.devicePixelRatioF()
        else:
            actual = harness.window._preview_canvas.view_zoom() * harness.window.devicePixelRatioF()
        assert actual == pytest.approx(.4)
    harness.assert_clean()


@pytest.mark.parametrize("mode,value", [("pixel", None), ("fit", None),
                                        ("percent", 143), ("magnification", 275)])
def test_new_main_window_restores_last_trial_and_display_preference(tmp_path, mode, value):
    profile = tmp_path / "restart-profile"
    with workflow.WorkflowHarness(profile) as first:
        first.enter()
        first.window._preview_display_binding.set_mode(mode, value)
        first.select(2)
        first.assert_clean()
    with workflow.WorkflowHarness(profile) as second:
        second.enter()
        assert second.hub.navigation.currentIndex() == 2
        page = second.hub.pages.currentWidget()
        assert page.display_binding.store.preferences.mode == mode
        assert page.display_choice.currentData() == mode
        assert page.canvas.display_mode == ("pixel" if mode == "magnification" else mode)
        if value is not None:
            assert getattr(page.display_binding.store.preferences, mode) == value
        second.assert_navigation()
        second.assert_clean()


def test_missing_frames_pause_trial_and_new_complete_frame_recovers(harness):
    harness.enter(frame=False)
    page = harness.select(1, frame=False)
    assert harness.window._capture_manager.current_capture_state() == "starting"
    assert page.source is not None
    page.source._last_frame_at -= 4
    page.source._check_arrivals()
    workflow.settle()
    assert page.controller.state == "unavailable"
    assert page.canvas.frame is None
    harness.backend.tick()
    assert page.controller.latest_frame is not None
    assert page.canvas.frame is not None
    assert harness.window._capture_manager.current_capture_state() == "streaming"
    harness.assert_navigation()
    harness.assert_clean()


def test_camera_error_is_inline_retry_works_and_stale_callbacks_are_ignored(harness):
    harness.enter()
    harness.select(1)
    manager = harness.window._capture_manager
    harness.backend.fail()
    assert manager.current_capture_state() == "faulted"
    assert not manager.is_preview_active()
    assert harness.hub.feedback.isVisible()
    assert "验证" in harness.hub.feedback.text()
    assert not harness.modal_messages
    harness.hub.capture_button.click()
    workflow.settle()
    harness.backend.tick()
    assert manager.current_capture_state() == "streaming"
    latest = manager.latest_preview_frame()
    stale = QImage(*harness.backend.size, QImage.Format.Format_RGB32)
    stale.fill(QColor("red"))
    harness.backend.tick(callback_index=0, image=stale)
    harness.backend.fail("过期错误必须忽略", callback_index=0)
    assert manager.latest_preview_frame().sequence == latest.sequence
    assert manager.latest_preview_frame().image == latest.image
    assert manager.current_capture_state() == "streaming"
    assert harness.backend.starts == 2
    harness.assert_navigation()
    harness.assert_clean()


def test_fast_reentry_waits_for_async_stop_then_restarts_once(tmp_path):
    backend = workflow.FakeCaptureBackend(delayed_stop=True)
    with workflow.WorkflowHarness(tmp_path / "async-profile", backend=backend) as harness:
        harness.enter()
        harness.select(1)
        harness.leave()
        assert harness.window._capture_manager.is_stopping()
        harness.enter(frame=False)
        assert backend.starts == 1
        assert harness.hub._restart_pending
        backend.finish_stop()
        assert backend.starts == 2
        backend.tick()
        assert harness.window._capture_manager.current_capture_state() == "streaming"
        assert harness.hub.pages.currentWidget().canvas.frame is not None
        harness.assert_navigation()
        harness.assert_clean()


def test_review_and_reports_keep_same_image_rectangle(harness):
    harness.enter()
    for index in (1, 2):
        page = harness.select(index)
        initial = page.image_scroll.geometry()
        assert page.sample_summary.isVisible() and page.ruler_choice.isVisible()
        for tab in (1, 2, 0):
            page.workflow_tabs.setCurrentIndex(tab)
            workflow.settle()
            assert page.image_scroll.geometry() == initial
            assert page.image_scroll.mapTo(harness.window, page.image_scroll.rect().bottomRight()).y() < harness.window.height()
    harness.assert_clean()


def test_digital_slide_entry_probes_no_real_motor_and_returns_to_measurement(harness):
    harness.window.digital_slide_action.trigger()
    workflow.settle()
    harness.backend.tick()
    assert harness.window._digital_slide_mode
    assert harness.window._slide_motion._serial is None
    assert not harness.window._slide_motion.enabled
    assert not harness.window._microview_preview_host.isVisible()
    harness.leave()
    assert not harness.window._digital_slide_mode
    harness.assert_clean()


def _pixel_grid():
    y, x = np.ogrid[:1080, :1920]
    values = ((x + y) % 2 * 255).astype(np.uint8)
    image = QImage(values.data, 1920, 1080, 1920, QImage.Format.Format_Grayscale8).convertToFormat(
        QImage.Format.Format_RGB32)
    painter = QPainter(image)
    for x, y, color in ((0, 0, "red"), (1896, 0, "green"), (0, 1056, "blue"), (1896, 1056, "yellow")):
        painter.fillRect(x, y, 24, 24, QColor(color))
    painter.end()
    return image


def test_slide_camera_display_modes_and_nine_anchors_without_another_frame(harness):
    from fdm.geometry import Point
    from fdm.live_experiment.calibration_store import save_ruler
    from fdm.live_experiment.controller import CalibrationSnapshot
    from fdm.ui.acquisition_alignment import ALIGNMENT_FACTORS
    from fdm.ui.display_preferences import viewport_key
    from fdm.ui.live_display import save_screen_ruler

    window = harness.window
    native = _pixel_grid()
    size = native.size().toTuple()
    source_id = "camera:" + harness.backend.device.id
    save_ruler(viewport_key(source_id, size), CalibrationSnapshot(.5, .5))
    save_screen_ruler(window.screen(), 125)
    window._toggle_digital_slide_mode(True)
    harness.backend.tick(image=native)
    canvas = window._preview_canvas
    binding = window._preview_display_binding
    sequence = window._preview_frame_serial
    assert binding._source == (source_id, size)
    with patch.object(window, "_perform_digital_slide_jog_step") as jog:
        for mode, physical_scale in (("fit", None), ("magnification", .4), ("percent", 1.25), ("pixel", 1)):
            with patch("fdm.ui.main_window.QInputDialog.getDouble", return_value=(125, True)), \
                    patch("fdm.ui.live_display.request_magnification", return_value=(200, 125)):
                choice = window._preview_display_choice
                choice.activated.emit(choice.findData(mode))
            workflow.settle()
            dpr = canvas.devicePixelRatioF()
            if physical_scale is not None:
                assert canvas.view_zoom() * dpr == pytest.approx(physical_scale)
                assert canvas._base_image_raster() == native
            else:
                assert canvas._base_image_raster().size().toTuple() == (1280, 720)
                assert "1280 × 720" in window._preview_status_label.text()
            assert window._preview_document.image_size == size
            for alignment, factors in ALIGNMENT_FACTORS.items():
                window._preview_alignment_button.direction_buttons[alignment].click()
                workflow.settle()
                origin = canvas.image_to_widget(Point(0, 0))
                expected = tuple((v - s * canvas.view_zoom()) * f for v, s, f in
                                 zip(canvas.size().toTuple(), size, factors))
                assert origin.toTuple() == pytest.approx(expected, abs=.51 / dpr)
                if mode == "pixel":
                    pixels = canvas.grab().toImage()
                    ox, oy = (round(value * dpr) for value in origin.toTuple())
                    sx = round((canvas.width() / 2 - origin.x()) * dpr)
                    sy = round((canvas.height() / 2 - origin.y()) * dpr)
                    for col in range(sx, sx + 4):
                        assert pixels.pixelColor(ox + col, oy + sy) == native.pixelColor(col, sy)
        assert window._preview_frame_serial == sequence
        assert window._latest_preview_frame == native
        jog.assert_not_called()
        position = QPointF(canvas.rect().center())
        event = QWheelEvent(position, position, QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(canvas, event)
        from fdm.services.motion_control import AXIS_Z, DIR_POS
        jog.assert_called_once_with(AXIS_Z, DIR_POS)
        assert binding.store.preferences.mode == "pixel"
        with patch.object(window, "_slide_acquisition_active", return_value=True):
            window._last_digital_slide_focus_wheel_at = 0
            QApplication.sendEvent(canvas, event)
            assert jog.call_count == 1
            assert canvas.view_zoom() * dpr == pytest.approx(1)
    harness.assert_clean()


def test_slide_fitted_pan_uses_native_zoom_and_center_between_frames(harness):
    window = harness.window
    harness.enter(frame=False)
    native = _pixel_grid()
    harness.backend.tick(image=native)
    binding, canvas = window._preview_display_binding, window._preview_canvas
    binding.set_mode("fit")
    binding.set_alignment("bottom_right")
    workflow.settle()
    window._toggle_digital_slide_mode(True)
    workflow.settle()
    serial = window._preview_frame_serial
    assert canvas._base_image_raster().size().toTuple() == (1280, 720)
    window._app_settings.digital_slide_preview_max_width = 0
    window._activate_app_settings(window._app_settings)
    assert canvas._base_image_raster() == native
    window._app_settings.digital_slide_preview_max_width = 960
    window._activate_app_settings(window._app_settings)
    workflow.settle()
    assert canvas._base_image_raster().width() == 960
    assert window._preview_frame_serial == serial
    fitted_zoom = canvas.view_zoom()
    center = canvas.rect().center()
    delta = QPoint(26, 38)
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=center)
    QTest.mouseMove(canvas, center + delta)
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=center + delta)
    workflow.settle()
    assert binding.store.preferences.alignment == "free"
    assert binding.store.preferences.percent == pytest.approx(fitted_zoom * canvas.devicePixelRatioF() * 100)
    assert canvas._base_image_raster() == native
    pan = canvas._pan
    harness.backend.tick(image=native)
    assert canvas.view_zoom() == pytest.approx(fitted_zoom)
    assert canvas._pan == pan
    window._toggle_digital_slide_mode(False)
    workflow.settle()
    assert canvas.view_zoom() == pytest.approx(fitted_zoom)
    # Change source geometry before the previous mode's queued center restore.
    binding.set_mode("percent", 275)
    resized = native.scaled(980, 520)
    window._on_live_preview_frame_ready(resized)
    workflow.settle()
    assert canvas.view_zoom() * canvas.devicePixelRatioF() == pytest.approx(2.75)
    assert canvas.widget_to_image(QPointF(canvas.width() / 2, canvas.height() / 2)).x == pytest.approx(490)
    binding.set_mode("magnification", 300)
    harness.backend.tick(image=resized)
    assert binding._source[1] == (980, 520)
    assert canvas.view_zoom() * canvas.devicePixelRatioF() == pytest.approx(1)
    assert binding.store.preferences.mode == "magnification"
    harness.assert_clean()


def test_no_device_entry_has_inline_guidance_and_retry(harness):
    harness.backend.available = False
    harness.enter(frame=False)
    assert not harness.window._capture_manager.is_preview_active()
    assert harness.hub.feedback.isVisible()
    assert not harness.modal_messages
    harness.backend.available = True
    harness.hub.capture_button.click()
    workflow.settle()
    harness.backend.tick()
    assert harness.window._capture_manager.is_preview_active()
    harness.assert_clean()


def test_empty_draft_closes_through_real_save_preflight_without_prompt(harness):
    harness.enter()
    harness.select(1)
    harness.select(2)
    harness.assert_clean()
    assert harness.window.close()
    workflow.settle()
    assert not harness.window.isVisible()
    assert not harness.modal_messages
    assert not harness.slot_errors


@pytest.mark.parametrize("size", [(1366, 768), (1280, 720), (1366, 690), (1280, 670)])
def test_nine_materials_remain_reachable_by_scrolling_at_small_logical_size(tmp_path, size):
    from fdm.live_experiment.models import MaterialSnapshot
    with workflow.WorkflowHarness(tmp_path / "nine-categories", size=size) as harness:
        harness.enter()
        page = harness.select(1)
        image_rectangle = page.image_scroll.geometry()
        page.session.set_materials([MaterialSnapshot(str(i), f"纤维类别 {i + 1}", 1.3) for i in range(9)])
        page._refresh_materials()
        workflow.settle()
        controls = page.category_scroll
        assert controls.verticalScrollBar().maximum() > 0
        for row in page.material_rows.values():
            controls.ensureWidgetVisible(row["select"])
            workflow.settle()
            assert row["select"].visibleRegion().contains(row["select"].rect())
        assert page.image_scroll.geometry() == image_rectangle
        harness.assert_size()
