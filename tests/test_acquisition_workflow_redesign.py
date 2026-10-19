"""Regression routes use MainWindow actions and a deterministic fake camera."""
from pathlib import Path
import importlib.util

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage
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
