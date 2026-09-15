import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import time
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from shiboken6 import isValid
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from fdm.services.contour_comparison import ContourAxis, ContourFrame, frame_from_rgba, load_comparison
from fdm.ui.contour_comparison_dialog import ContourComparisonDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, condition, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        QTest.qWait(5)
    raise AssertionError("Qt task did not finish")


def settle(app, dialog):
    wait(app, lambda: not dialog._tasks.busy and not dialog._timer.isActive())
    app.processEvents()


def sample(label="before", dx=0, size=(260, 200)):
    rgba = np.full((*size, 4), 250, np.uint8)
    rgba[20:240, 40+dx:160+dx, :3] = (35, 75, 120)
    rgba[90:240, 87+dx:113+dx, :3] = 250
    f = frame_from_rgba(rgba, label, mm_per_pixel=.5)
    return replace(f, axis=ContourAxis((100, 20), (100, 240)))


@pytest.fixture
def dialog(app):
    d = ContourComparisonDialog()
    d.show()
    app.processEvents()
    yield d
    if d._tasks.busy:
        d._tasks.cancel()
        settle(app, d)
    d.dirty = False
    d.close()
    app.processEvents()


def populate(app, d):
    d._replace_frame(0, lambda token: sample(), "load")
    settle(app, d)
    d._replace_frame(1, lambda token: sample("after", 2), "load")
    settle(app, d)
    assert d.result is not None, d.status.text()


def test_loaded_fixed_camera_uses_before_axis_and_reports_displacement(app, dialog):
    populate(app, dialog)
    assert dialog.frames[0].axis == dialog.frames[1].axis
    assert dialog.result.sections[0].left_change == -1
    assert dialog.result.sections[0].right_change == 1
    assert dialog.export_button.isEnabled()
    dialog.tabs.setCurrentIndex(1)
    app.processEvents()
    dialog.table.selectRow(5)
    assert dialog.overlay.height_line == dialog.result.sections[5].height


def test_manual_brush_and_polygon_keep_native_geometry_and_undo_redo(app, dialog):
    populate(app, dialog)
    original = dialog.frames[1]
    canvas = dialog.canvases[1]
    dialog._set_mode("brush_remove")
    p = canvas.transform().map(QPointF(140, 60)).toPoint()
    q = canvas.transform().map(QPointF(130, 75)).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=p)
    QTest.mouseMove(canvas, q)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=q)
    settle(app, dialog)
    assert not dialog.frames[1].mask[68, 135]
    assert original.mask[68, 135]
    assert dialog.frames[1].axis == original.axis
    dialog.undo()
    settle(app, dialog)
    np.testing.assert_array_equal(dialog.frames[1].mask, original.mask)
    dialog.redo()
    settle(app, dialog)
    assert not dialog.frames[1].mask[68, 135]
    dialog._set_mode("polygon_add")
    for point in ((130, 50), (155, 50), (155, 80), (130, 80)):
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.transform().map(QPointF(*point)).toPoint())
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    settle(app, dialog)
    assert dialog.frames[1].mask[68, 135]


def test_axis_and_calibration_from_either_shared_pane_and_independent_undo(app, dialog):
    populate(app, dialog)
    dialog._gesture(1, "axis", [(102, 20), (102, 80)])
    settle(app, dialog)
    assert dialog.frames[0].axis == dialog.frames[1].axis
    assert dialog.frames[0].axis.origin == (102, 20)
    with patch("fdm.ui.contour_comparison_dialog.QInputDialog.getDouble", return_value=(20, True)):
        dialog._gesture(1, "calibrate", [(10, 10), (110, 10)])
    settle(app, dialog)
    assert dialog.frames[0].mm_per_pixel == .2
    assert dialog.frames[1].mm_per_pixel == .2
    dialog.same_capture.setChecked(False)
    settle(app, dialog)
    dialog._gesture(1, "axis", [(105, 20), (105, 100)])
    settle(app, dialog)
    assert dialog.frames[0].axis.origin == (102, 20)
    assert dialog.frames[1].axis.origin == (105, 20)
    dialog.undo()
    settle(app, dialog)
    assert dialog.frames[1].axis.origin == (102, 20)


def test_save_reopen_and_exports_are_consistent_snapshots(app, dialog, tmp_path):
    populate(app, dialog)
    dialog._gesture(1, "polygon_remove", [(120, 30), (160, 30), (160, 80), (120, 80)])
    settle(app, dialog)
    mask = dialog.frames[1].mask.copy()
    path = tmp_path / "paired.fdmcompare"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
        dialog._save()
    settle(app, dialog)
    assert not dialog.dirty
    _, after, _, _ = load_comparison(path)
    np.testing.assert_array_equal(after.mask, mask)
    dialog.undo()
    settle(app, dialog)
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard), patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
        dialog._open_session()
    settle(app, dialog)
    np.testing.assert_array_equal(dialog.frames[1].mask, mask)
    assert not dialog.dirty
    for method, name in ((dialog._export, "result.xlsx"), (dialog._export_overlay, "overlay.png")):
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(tmp_path/name), "")):
            method()
        settle(app, dialog)
        assert (tmp_path/name).stat().st_size > 500
    from PIL import Image
    with Image.open(tmp_path / "overlay.png") as image:
        assert image.info["Before"] == dialog.frames[0].label
        assert image.info["ComparisonUnit"] == "mm"
        assert "同高度" in image.info["Notes"]


def test_different_dimensions_disable_shared_coordinates(app, dialog):
    dialog._replace_frame(0, lambda token: sample(), "load")
    settle(app, dialog)
    other = sample("larger", size=(280, 220))
    dialog._replace_frame(1, lambda token: other, "load")
    settle(app, dialog)
    assert not dialog.same_capture.isChecked()
    dialog.same_capture.setChecked(True)
    assert not dialog.same_capture.isChecked()
    assert "尺寸不同" in dialog.status.text()


def test_cancel_late_work_never_installs_and_canvas_pan_remains_responsive(app, dialog):
    populate(app, dialog)
    before = dialog.frames[0]
    started = Event()
    def work(token):
        started.set()
        token.wait(5)
        token.raise_if_cancelled()
        return sample("should-not-install")
    dialog._replace_frame(0, work, "cancel test")
    wait(app, started.is_set)
    canvas = dialog.canvases[0]
    old = QPointF(canvas.offset)
    QTest.mousePress(canvas, Qt.MouseButton.RightButton, pos=QPoint(100, 100))
    QTest.mouseMove(canvas, QPoint(145, 155))
    QTest.mouseRelease(canvas, Qt.MouseButton.RightButton, pos=QPoint(145, 155))
    assert canvas.offset != old
    dialog._tasks.cancel()
    settle(app, dialog)
    assert dialog.frames[0] is before
    assert "已取消" in dialog.status.text()


def test_close_running_task_waits_for_worker_without_destroying_signal_carrier(app):
    d = ContourComparisonDialog()
    d.show()
    started = Event()
    def work(token):
        started.set()
        token.wait(5)
        token.raise_if_cancelled()
    d._start("cancel-on-close", work, lambda result: None)
    wait(app, started.is_set)
    assert not d.close()
    wait(app, lambda: not isValid(d) or not d.isVisible())
    app.processEvents()


def test_escape_cancels_pending_polygon_and_does_not_close_session(app, dialog):
    populate(app, dialog)
    canvas = dialog.canvases[0]
    dialog._set_mode("polygon_remove")
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.transform().map(QPointF(60, 60)).toPoint())
    assert canvas.has_gesture()
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert not canvas.has_gesture()
    assert dialog.isVisible()


def test_pending_polygon_cannot_be_silently_saved_or_exported(app, dialog):
    populate(app, dialog)
    canvas = dialog.canvases[0]
    dialog._set_mode("polygon_remove")
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.transform().map(QPointF(60, 60)).toPoint())
    with patch.object(QFileDialog, "getSaveFileName") as picker:
        dialog._save()
        dialog._export()
        dialog._export_overlay()
    picker.assert_not_called()
    assert "未完成" in dialog.status.text()


def test_main_window_entry_is_independent_and_source_snapshot_outlives_tab(app):
    from fdm.cancellation import CancellationTokenSource
    from fdm.models import Calibration, ImageDocument, new_id
    from fdm.raster import RasterPixelType, RasterPlane
    from fdm.services.raster_io import raster_plane_to_qimage
    from fdm.settings import AppSettings
    from fdm.ui.main_window import MainWindow

    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings(theme_mode="dark")), patch("fdm.ui.main_window.AppSettingsIO.save"):
        window = MainWindow()
        try:
            f = sample()
            plane = RasterPlane(f.rgba.shape[1], f.rgba.shape[0], RasterPixelType.RGB8, f.rgba[:, :, :3].tobytes())
            doc = ImageDocument(id=new_id("image"), path="/tmp/contour-source.png", image_size=(plane.width, plane.height))
            doc.calibration = Calibration("manual", 2, "mm", "ruler")
            doc.initialize_runtime_state()
            doc.mark_session_saved()
            doc.mark_calibration_saved()
            window._mount_document(doc, raster_plane_to_qimage(plane), tooltip=doc.path, raster_plane=plane)
            window.contour_comparison_action.trigger()
            app.processEvents()
            dialog = window._contour_comparison_dialog
            assert dialog is not None and dialog.isVisible()
            window._open_contour_comparison()
            assert window._contour_comparison_dialog is dialog
            assert window._contour_comparison_sources()[0][0] == doc.id
            loader = window._contour_comparison_source_loader(doc.id)
            window._reset_workspace()
            loaded = loader(CancellationTokenSource().token)
            assert loaded.mm_per_pixel == .5
            assert loaded.mask.any()
            np.testing.assert_array_equal(loaded.rgba[:, :, :3], f.rgba[:, :, :3])
            assert window.project.documents == []
            assert not dialog.dirty
            dialog.close()
            app.processEvents()
        finally:
            window._reset_workspace()
            window.close()
            app.processEvents()


def test_undo_during_recalculation_is_applied_after_cancellation(app, dialog):
    populate(app, dialog)
    dialog._set_geometry(0, axis=ContourAxis((102, 20), (102, 200)))
    dialog._timer.stop()
    started = Event()
    def work(token):
        started.set()
        token.wait(5)
        token.raise_if_cancelled()
    dialog._start("计算轮廓变化", work, lambda _: None)
    wait(app, started.is_set)
    dialog.undo()
    settle(app, dialog)
    assert dialog.frames[0].axis.origin == (100, 20)
    assert dialog.frames[1].axis.origin == (100, 20)
    assert dialog.result is not None


def test_sampling_changes_undo_and_empty_preferences_do_not_require_save(app, dialog):
    dialog.step.setValue(5)
    assert not dialog.dirty
    populate(app, dialog)
    dialog.step.setValue(2)
    settle(app, dialog)
    dialog.undo()
    settle(app, dialog)
    assert dialog.step.value() == 5
    assert dialog.result.step == 5


def test_overlay_native_mask_transform_equals_measurement_frame(app, dialog):
    populate(app, dialog)
    dialog.same_capture.setChecked(False)
    settle(app, dialog)
    dialog._set_geometry(1, axis=ContourAxis((105, 45), (35, 170)))
    settle(app, dialog)
    points = np.array([[10.5, 20.5], [120, 220], [90, 100]])
    for frame, (_mask, transform) in zip(dialog.frames, dialog.overlay.rasters):
        display_points = np.array([[transform.map(QPointF(*p)).x(), transform.map(QPointF(*p)).y()] for p in points])
        np.testing.assert_allclose(display_points, frame.axis.to_world(points, frame.scale))
    before_result = dialog.result
    dialog.coverage_visible.setChecked(False)
    assert not any(canvas.show_mask for canvas in dialog.canvases)
    assert dialog.result is before_result


def test_wand_click_prompt_refinement_and_manual_edit_invalidate_only_that_source(app, dialog):
    from fdm.services.contour_comparison_wand import apply_wand_mask
    populate(app, dialog)
    calls = []
    def predict(frame, positive, negative, *, operation, base_mask, token):
        calls.append((positive, negative, operation, base_mask))
        mask = np.zeros_like(frame.mask)
        mask[30:80, 20:75 if not negative else 55] = True
        return apply_wand_mask(frame, mask, operation=operation, base_mask=base_mask)
    dialog._wand_service.predict = predict
    dialog._set_mode('wand')
    canvas = dialog.canvases[0]
    point = canvas.transform().map(QPointF(45, 50)).toPoint()
    old = dialog.frames[0]
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point)
    settle(app, dialog)
    first = dialog.frames[0]
    assert first.mask[50, 30] and not first.mask[90, 140]
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, canvas.transform().map(QPointF(65, 50)).toPoint())
    settle(app, dialog)
    assert not dialog.frames[0].mask[50, 65]
    assert len(calls[-1][0]) == 1 and len(calls[-1][1]) == 1
    assert calls[-1][3] is old.mask
    assert len(canvas.wand_prompts) == 2
    # Refining the other image cannot silently discard this pane's point set.
    dialog._gesture(1, 'wand', [(45, 50)])
    settle(app, dialog)
    assert len(canvas.wand_prompts) == 2
    dialog._gesture(0, 'brush_add', [(140, 150)])
    settle(app, dialog)
    assert not canvas.wand_prompts
    assert dialog.canvases[1].wand_prompts
    dialog.undo()
    settle(app, dialog)
    assert not dialog.frames[0].mask[150, 140]
    # Prompt reset never mutates the saved contour.
    before = dialog.frames[0]
    dialog._reset_wand()
    assert dialog.frames[0] is before


@pytest.mark.parametrize('operation', ['add', 'remove'])
def test_wand_refinement_with_checkbox_and_undo_uses_session_base(app, dialog, operation):
    from fdm.services.contour_comparison_wand import apply_wand_mask
    populate(app, dialog)
    base = dialog.frames[0]
    def predict(frame, positive, negative, *, operation, base_mask, token):
        mask = np.zeros_like(frame.mask)
        mask[30:80, 20:75 if not negative else 55] = True
        return apply_wand_mask(frame, mask, operation=operation, base_mask=base_mask)
    dialog._wand_service.predict = predict
    dialog.wand_operation.setCurrentIndex(dialog.wand_operation.findData(operation))
    dialog._gesture(0, 'wand', [(45, 50)])
    settle(app, dialog)
    first = dialog.frames[0]
    dialog.wand_negative.setChecked(True)
    dialog._gesture(0, 'wand', [(65, 50)])
    settle(app, dialog)
    assert dialog.wand_negative.isChecked()
    selected = np.zeros_like(base.mask); selected[30:80, 20:55] = True
    expected = base.mask | selected if operation == 'add' else base.mask & ~selected
    np.testing.assert_array_equal(dialog.frames[0].mask, expected)
    dialog.undo()
    settle(app, dialog)
    np.testing.assert_array_equal(dialog.frames[0].mask, first.mask)
    assert not dialog.canvases[0].wand_prompts


def test_wand_pending_cancel_and_negative_without_target_are_safe(app, dialog):
    populate(app, dialog)
    started = Event()
    def predict(frame, *args, token, **kwargs):
        started.set(); token.wait(5); token.raise_if_cancelled()
    dialog._wand_service.predict = predict
    old = dialog.frames[0]
    dialog._gesture(0, 'wand_negative', [(45, 50)])
    assert not dialog._tasks.busy and '请先' in dialog.status.text()
    dialog._gesture(0, 'wand', [(45, 50)])
    wait(app, started.is_set)
    dialog.undo()  # Ctrl+Z cancels the accepted pending inference.
    settle(app, dialog)
    assert dialog.frames[0] is old and dialog._wand_sessions[0] is None


@pytest.mark.parametrize('method, picker', [('_import_file', 'getOpenFileName'), ('_open_session', 'getOpenFileName'), ('_save', 'getSaveFileName'), ('_export', 'getSaveFileName'), ('_export_overlay', 'getSaveFileName')])
def test_all_cancelled_file_pickers_restore_workspace_without_changing_it(app, dialog, method, picker):
    from PySide6.QtWidgets import QWidget
    populate(app, dialog)
    before, result, history = tuple(dialog.frames), dialog.result, len(dialog._history)
    background = QWidget(); background.show()
    def cancel(*args):
        assert args[0] is dialog
        assert dialog._picker_depth == 1
        background.activateWindow()
        app.processEvents()
        dialog.reject()  # A native picker Escape must not dismiss the owner.
        return '', ''
    try:
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard), patch.object(QFileDialog, picker, side_effect=cancel):
            getattr(dialog, method)(0) if method == '_import_file' else getattr(dialog, method)()
        app.processEvents()
        assert dialog.isVisible() and dialog._picker_depth == 0
        assert app.activeWindow() is dialog
        assert tuple(dialog.frames) == before and dialog.result is result
        assert len(dialog._history) == history and dialog.dirty
    finally:
        background.close()


def test_result_click_and_height_controls_locate_same_exact_section(app, dialog):
    populate(app, dialog)
    dialog.tabs.setCurrentIndex(1); app.processEvents()
    overlay = dialog.overlay
    row = dialog.result.sections[3]
    position = overlay.transform().map(QPointF(0, row.height)).toPoint()
    QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=position)
    assert dialog.table.currentIndex().row() == 3
    assert overlay.section is row and overlay.height_line == row.height
    assert dialog.section_height.value() == row.height
    dialog._step_section(1)
    assert overlay.section is dialog.result.sections[4]
    dialog.section_height.setValue(dialog.result.sections[0].height)
    assert not dialog.previous_section.isEnabled()
    snapshot = dialog.result
    for background in ('after', 'none', 'before'):
        dialog.result_background.setCurrentIndex(dialog.result_background.findData(background))
        assert overlay.background == background and dialog.result is snapshot
    dialog.more_columns.setChecked(True)
    assert not dialog.table.isColumnHidden(4)
    dialog.more_columns.setChecked(False)
    assert dialog.table.isColumnHidden(4)


def test_result_missing_height_has_no_invented_zero_and_exports_selected_location(app, dialog, tmp_path):
    from PIL import Image
    populate(app, dialog)
    frame = dialog.frames[1]
    mask = frame.mask.copy(); mask[:80] = False
    dialog._replace_frame(1, lambda token: replace(frame, mask=mask), 'edit')
    settle(app, dialog)
    dialog.table.selectRow(0)
    row = dialog.overlay.section
    assert row.after_intervals == () and row.left_change is None
    assert dialog.overlay.change_text(row.left_change, 'mm') == '此侧无法比较'
    assert '仅处理前' in dialog.section_label.text()
    dialog.result_background.setCurrentIndex(dialog.result_background.findData('after'))
    path = tmp_path / 'located.png'
    with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(path), '')):
        dialog._export_overlay()
    settle(app, dialog)
    with Image.open(path) as image:
        assert float(image.info['SelectedHeight']) == row.height
        assert image.info['Background'] == 'after'
    assert dialog.overlay.section is row


def test_photo_and_mask_share_native_mapping_for_independent_rotated_calibration(app, dialog):
    populate(app, dialog)
    dialog.same_capture.setChecked(False); settle(app, dialog)
    dialog._set_geometry(1, axis=ContourAxis((140, 80), (60, 190)), mm_per_pixel=.7)
    settle(app, dialog)
    for frame, (image, transform), (mask, mask_transform) in zip(dialog.frames, dialog.overlay.photos, dialog.overlay.rasters):
        assert transform == mask_transform
        assert image.width() == frame.mask.shape[1] and mask.height() == frame.mask.shape[0]
        point = QPointF(50.5, 120.25)
        shown = transform.map(point)
        np.testing.assert_allclose((shown.x(), shown.y()), frame.axis.to_world(np.array([[point.x(), point.y()]]), frame.scale)[0])
    dialog.tabs.setCurrentIndex(1); app.processEvents()
    for background in ('before', 'after', 'none'):
        dialog.overlay.set_background(background)
        assert not dialog.overlay.grab().isNull()


def test_file_picker_defers_pending_comparison_until_return(app, dialog):
    populate(app, dialog)
    dialog._set_geometry(0, axis=ContourAxis((102, 20), (102, 200)))
    assert dialog._timer.isActive()
    def cancel(*args):
        assert not dialog._timer.isActive()
        QTest.qWait(250)
        assert not dialog._tasks.busy
        return '', ''
    with patch.object(QFileDialog, 'getOpenFileName', side_effect=cancel):
        dialog._import_file(0)
    assert dialog._timer.isActive()
    settle(app, dialog)
    assert dialog.result is not None


def test_escape_finishes_wand_points_preserving_applied_mask(app, dialog):
    from fdm.services.contour_comparison_wand import apply_wand_mask
    populate(app, dialog)
    def predict(frame, *args, **kwargs):
        selected = np.zeros_like(frame.mask); selected[20:60, 35:90] = True
        return apply_wand_mask(frame, selected)
    dialog._wand_service.predict = predict
    dialog._set_mode('wand')
    dialog._gesture(0, 'wand', [(45, 50)]); settle(app, dialog)
    frame = dialog.frames[0]
    QTest.keyClick(dialog.canvases[0], Qt.Key.Key_Escape)
    assert dialog.isVisible() and dialog.frames[0] is frame
    assert not dialog.canvases[0].wand_prompts and dialog._wand_sessions[0] is None


def test_cancellation_wins_over_already_queued_worker_result(app):
    from fdm.ui.contour_comparison_tasks import ContourTaskController
    tasks = ContourTaskController()
    results = []
    tasks.finished.connect(lambda value, error, cancelled: results.append((value, error, cancelled)))
    tasks.start(lambda token: 'late result')
    assert tasks.wait_for_done(3000)
    tasks.cancel()  # Finished in the worker, still queued on the GUI thread.
    wait(app, lambda: bool(results))
    assert results == [(None, '', True)]


def test_compact_wand_layout_keeps_calibration_below_each_canvas(app, dialog):
    populate(app, dialog)
    dialog.resize(860, 640); dialog._set_mode('wand'); app.processEvents()
    for canvas, label in zip(dialog.canvases, dialog.scale_labels):
        assert canvas.geometry().bottom() < label.geometry().top()
    assert dialog.size().width() == 860 and dialog.size().height() == 640


def test_pending_comparison_does_not_start_inside_calibration_prompt(app, dialog):
    from PySide6.QtWidgets import QDialog
    populate(app, dialog)
    dialog._set_geometry(0, axis=ContourAxis((102, 20), (102, 200)))
    def get_distance(*args):
        modal = QDialog(dialog); modal.setModal(True); modal.show()
        app.processEvents(); QTest.qWait(300)
        assert not dialog._tasks.busy
        modal.close(); app.processEvents()
        return 20, True
    with patch('fdm.ui.contour_comparison_dialog.QInputDialog.getDouble', side_effect=get_distance):
        dialog._gesture(0, 'calibrate', [(10, 10), (110, 10)])
    settle(app, dialog)
    assert dialog.result is not None and dialog.frames[0].mm_per_pixel == .2
    assert dialog._result_revision == dialog._revision


def test_empty_photo_buttons_cancel_without_hiding_workspace(app, dialog):
    for i, canvas in enumerate(dialog.canvases):
        assert canvas.import_button.isVisible()
        with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")) as choose:
            QTest.mouseClick(canvas.import_button, Qt.MouseButton.LeftButton)
        assert choose.call_count == 1
        assert ("处理前" if i == 0 else "处理后") in choose.call_args.kwargs.get("caption", choose.call_args.args[1])
        assert dialog.isVisible() and not dialog.dirty
        assert dialog.frames == [None, None]
    populate(app, dialog)
    assert all(not canvas.import_button.isVisible() for canvas in dialog.canvases)


@pytest.mark.parametrize("size", [(860, 640), (1280, 880)])
@pytest.mark.parametrize("theme", ["light", "dark", "system"])
def test_switching_tools_never_moves_photos_or_clips_toolbar(app, dialog, size, theme):
    from fdm.ui.theme import apply_application_theme
    apply_application_theme(app, theme)
    populate(app, dialog)
    dialog.resize(*size)
    app.processEvents()
    baseline = [canvas.geometry() for canvas in dialog.canvases]
    for mode in dialog.tools:
        dialog._set_mode(mode)
        app.processEvents()
        assert [canvas.geometry() for canvas in dialog.canvases] == baseline, mode
        for button in dialog.tools.values():
            assert button.isVisible() and button.width() >= button.sizeHint().width(), (theme, size, mode, button.text())
            assert dialog.rect().contains(button.mapTo(dialog, button.rect().bottomRight()))
    # Filename and quality notices must not resize or move either photo.
    for label in dialog.name_labels:
        label.setText("非常长的中文照片文件名称" * 50)
    dialog.warning_label.setText("轮廓需要复核。" * 80)
    app.processEvents()
    assert [canvas.geometry() for canvas in dialog.canvases] == baseline
    assert (dialog.width(), dialog.height()) == size


def test_active_photo_highlight_does_not_tint_its_image(app, dialog):
    populate(app, dialog)
    canvas = dialog.canvases[0]
    p = canvas.transform().map(QPointF(60, 60)).toPoint()
    dialog._activate(0)
    selected = canvas.grab().toImage()
    dialog._activate(1)
    other = canvas.grab().toImage()
    # Selection is a border/text change; the specimen's displayed pixel must
    # not inherit the reference-label painter's dark background brush.
    assert selected.pixelColor(p) == other.pixelColor(p)
    assert "正在编辑" in dialog.active_labels[1].text()


def test_canvas_shortcuts_and_temporary_pan_keep_geometry_intact(app, dialog):
    populate(app, dialog)
    canvas = dialog.canvases[1]
    original = dialog.frames[1].mask
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))
    QTest.keyClick(canvas, Qt.Key.Key_B)
    assert dialog.tools["brush_add"].isChecked()
    QTest.keyClick(canvas, Qt.Key.Key_X)
    assert dialog.tools["brush_remove"].isChecked()
    radius = dialog.brush.value()
    QTest.keyClick(canvas, Qt.Key.Key_BracketRight)
    assert dialog.brush.value() > radius
    old_offset = QPointF(canvas.offset)
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseMove(canvas, QPoint(130, 125))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(130, 125))
    QTest.keyRelease(canvas, Qt.Key.Key_Space)
    assert canvas.offset != old_offset and not canvas.has_gesture()
    assert dialog.frames[1].mask is original and not dialog._tasks.busy
    QTest.keyClick(canvas, Qt.Key.Key_F)
    QTest.keyClick(canvas, Qt.Key.Key_P)
    for point in ((60, 50), (65, 70)):
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.transform().map(QPointF(*point)).toPoint())
    assert len(canvas.points) == 2
    QTest.keyClick(canvas, Qt.Key.Key_Backspace)
    assert len(canvas.points) == 1
    # A shortcut must not silently switch/discard an unfinished polygon.
    QTest.keyClick(canvas, Qt.Key.Key_X)
    assert len(canvas.points) == 1 and canvas.mode == "polygon_add"
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert dialog.frames[1].mask is original


def test_result_locator_and_sampling_stay_linked_without_editing_geometry(app, dialog):
    populate(app, dialog)
    dialog.tabs.setCurrentIndex(1)
    app.processEvents()
    frames = tuple(dialog.frames)
    result = dialog.result
    dialog.dirty = False
    dialog.section_slider.setValue(2)
    assert dialog.overlay.section is result.sections[2]
    assert dialog.table.currentIndex().row() == 2
    assert dialog.section_height.value() == result.sections[2].height
    assert dialog.position_label.text() == f"3 / {len(result.sections)}"
    QTest.keyClick(dialog.overlay, Qt.Key.Key_Down)
    assert dialog.section_slider.value() == 3
    dialog.records_visible.setChecked(False)
    dialog.result_background.setCurrentIndex(1)
    dialog.coverage_visible.setChecked(False)
    assert dialog.result is result and not dialog.dirty
    assert not dialog.records_panel.isVisible()
    dialog.records_visible.setChecked(True)
    dialog.result_step.setValue(5)
    settle(app, dialog)
    assert dialog.result.step == dialog.step.value() == 5
    assert dialog.result_step.suffix() == " mm"
    assert all(old is new for old, new in zip(frames, dialog.frames))
    dialog.undo()
    settle(app, dialog)
    assert dialog.step.value() == dialog.result_step.value() == 10


def test_reference_badge_locates_missing_calibration_and_independent_axis(app, dialog):
    for i in range(2):
        dialog._replace_frame(i, lambda token, i=i: replace(sample(str(i)), mm_per_pixel=None), "load")
        settle(app, dialog)
    dialog.tabs.setCurrentIndex(1)
    dialog.quality_badge.click()
    assert dialog.tabs.currentIndex() == 0 and dialog.tools["calibrate"].isChecked()
    assert dialog.active == 0 and "仅像素" in dialog.quality_badge.text()
    dialog.same_capture.setChecked(False)
    settle(app, dialog)
    dialog._set_geometry(0, mm_per_pixel=.5)
    settle(app, dialog)
    assert "不一致" in dialog.quality_badge.text()
    dialog.quality_badge.click()
    assert dialog.active == 1
    dialog._set_geometry(1, mm_per_pixel=.5)
    settle(app, dialog)
    dialog.quality_badge.click()
    assert dialog.tools["axis"].isChecked()
    first_axis = dialog.frames[0].axis
    dialog._gesture(1, "axis", [(102, 20), (102, 200)])
    settle(app, dialog)
    assert dialog.frames[0].axis == first_axis
    assert dialog.frames[1].axis.origin == (102, 20)
    assert "分别设置" in dialog.reference_hint.text()


def test_outline_edit_preserves_manually_positioned_result_view(app, dialog):
    populate(app, dialog)
    dialog.overlay.zoom = 2
    dialog.overlay.offset = QPointF(80, 90)
    dialog.overlay._auto_fit = False
    dialog._gesture(1, "polygon_remove", [(140, 60), (150, 60), (150, 80), (140, 80)])
    settle(app, dialog)
    assert dialog.overlay.zoom == 2 and dialog.overlay.offset == QPointF(80, 90)
    # A different coordinate frame must be fitted anew, not silently aligned
    # using an old viewport expressed in another reference system.
    dialog._set_geometry(0, axis=ContourAxis((105, 20), (105, 200)))
    settle(app, dialog)
    assert dialog.overlay._auto_fit


def test_narrow_results_keep_photo_space_and_show_unavailable_values(app, dialog):
    populate(app, dialog)
    frame = dialog.frames[1]
    mask = frame.mask.copy()
    mask[:80] = False
    dialog._replace_frame(1, lambda token: replace(frame, mask=mask), "edit")
    settle(app, dialog)
    dialog.resize(860, 640)
    dialog.tabs.setCurrentIndex(1)
    app.processEvents()
    dialog.section_slider.setValue(0)
    assert dialog.table_model.data(dialog.table_model.index(0, 1)) == "—"
    assert "仅处理前" in dialog.table_model.data(dialog.table_model.index(0, 1), Qt.ItemDataRole.ToolTipRole)
    top, bottom = dialog.overlay._plot_layout(dialog.overlay.size())
    assert dialog.overlay.height() - top - bottom >= 120
    assert all(not card.description.isVisible() for card in dialog.metrics.values())
    assert dialog.table.horizontalScrollBar().maximum() == 0
    # The dialog's default buttons must not consume polygon Enter gestures.
    assert not any(button.isDefault() for button in dialog.findChildren(QPushButton))
