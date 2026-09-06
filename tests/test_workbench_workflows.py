from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, new_id
from fdm.settings import AppSettings, RawRecordTemplate
from fdm.services.export_service import ExportImageRenderMode
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.main_window import MainWindow
from fdm.ui.canvas import AreaEditOperationMode, MagicSegmentOperationMode, MagicSegmentSubtractInputMode
from fdm.ui.workbench_controls import CommandSearchDialog


@pytest.fixture
def window(desktop_application):
    app = desktop_application
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=AppSettings()), \
         patch("fdm.ui.main_window.AppSettingsIO.save"):
        win = MainWindow()
        win.resize(1093, 576)
        win.show()
        win.activateWindow()
        app.processEvents()
        yield win
        win._preview_active = False
        win._digital_slide_mode = False
        with patch.object(win, "_confirm_close_documents", return_value=True):
            win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def mount(window, number, *, calibrated=True):
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.gray)
    document = ImageDocument(
        id=new_id("image"), path=f"/virtual/电镜样品-{number:02d}.png",
        image_size=(320, 240), source_type="project_asset",
        calibration=Calibration(mode="preset", pixels_per_unit=4, unit="um", source_label="20×") if calibrated else None,
    )
    document.initialize_runtime_state()
    cotton = document.create_group(color="#228877", label="棉")
    document.create_group(color="#AA7722", label="粘纤")
    measurement = Measurement(
        id=new_id("measurement"), image_id=document.id, fiber_group_id=cotton.id,
        mode="manual", line_px=Line(Point(20, 30), Point(100, 30)),
        confidence=1, status="manual",
    )
    document.add_measurement(measurement)
    window._add_loaded_document(ImageLoadRequest(path=document.path, document=document), image)
    return document


def test_twelve_images_switch_without_losing_snap_or_calibration_state(window):
    documents = [mount(window, i, calibrated=i % 2 == 0) for i in range(12)]
    window.set_tool_mode("snap")
    window._activate_document_index(0)
    for index, document in enumerate(documents):
        assert window.current_document() is document
        assert window._measurement_context_bar.documents.currentIndex() == index
        assert window.toggle_edge_snap_action.isChecked()
        assert window._tool_mode == "snap"
        assert window._measurement_context_bar.calibrationButton.property("uncalibrated") == (index % 2 == 1)
        window.next_image_action.trigger()
    assert window.current_document() is documents[-1]
    assert not window.next_image_action.isEnabled()
    window.previous_image_action.trigger()
    assert window.current_document() is documents[-2]
    window.toggle_edge_snap_action.trigger()
    assert window._tool_mode == "manual"
    assert window.current_canvas().hasFocus()


def test_category_shortcuts_and_quick_selector_only_affect_new_measurements(window):
    document = mount(window, 1)
    original = document.measurements[0]
    group1, group2 = document.sorted_groups()
    document.select_measurement(original.id)
    window._populate_measurement_table(document)
    window._focus_current_canvas()
    QTest.keyClick(window.current_canvas(), Qt.Key.Key_2)
    assert document.active_group_id == group2.id
    assert original.fiber_group_id == group1.id
    assert window._measurement_context_bar.groups.currentData() == group2.id
    window._measurement_context_bar.groupActivated.emit(group1.id)
    assert document.active_group_id == group1.id
    assert window.current_canvas().hasFocus()
    window._current_measurement_summary.groupChangeRequested.emit(original.id, group2.id)
    assert original.fiber_group_id == group2.id
    assert document.active_group_id == group1.id


def test_missing_scale_warning_survives_hidden_panels_and_tool_switch(window):
    mount(window, 1, calibrated=False)
    button = window._measurement_context_bar.calibrationButton
    for mode in ("manual", "snap", "polygon_area", "freehand_area"):
        window.set_tool_mode(mode)
        window._inspector_dock.hide()
        window._project_dock.hide()
        QApplication.processEvents()
        assert button.isVisible()
        assert button.property("uncalibrated")
        assert "px / px²" in button.text()
        assert window.rect().contains(button.mapTo(window, button.rect().bottomRight()))


def test_empty_start_does_not_show_measurement_warning(window):
    assert window._center_stack.currentWidget() is window._welcome_panel
    assert not window._measurement_context_bar.calibrationButton.property("uncalibrated")
    assert not window.previous_image_action.isEnabled()


def test_narrow_area_workspace_keeps_image_controls_and_scale_warning_visible(window):
    mount(window, 1, calibrated=False)
    window.set_tool_mode("polygon_area")
    window._set_workbench_presentation("focus")
    window.resize(692, 755)
    for _ in range(3):
        QApplication.processEvents()
    bar = window._measurement_context_bar
    assert window.width() == 692
    assert bar._compact
    for control in (bar.previousButton, bar.documents, bar.nextButton, bar.groups,
                    bar.calibrationButton, *bar.areaButtons):
        assert control.isVisible()
        assert window.rect().contains(control.mapTo(window, control.rect().topLeft()))
        assert window.rect().contains(control.mapTo(window, control.rect().bottomRight()))
    assert bar.calibrationButton.property("uncalibrated")


def test_template_and_overlay_commands_reuse_export_options(window):
    mount(window, 1)
    window._app_settings.raw_record_templates = [RawRecordTemplate(name="原始记录", path="/virtual/record.xlsx", rules=[])]
    with patch.object(window, "export_results") as export:
        window.export_template_action.trigger()
        preset = export.call_args.args[0]
        assert preset.include_excel
        assert preset.raw_record_template_path == "/virtual/record.xlsx"
        assert not preset.include_measurement_overlay
        window.export_overlay_action.trigger()
        assert export.call_args.args[0].include_measurement_overlay


def test_capture_commands_stay_visible_when_compact_inspector_is_hidden(window):
    window._digital_slide_mode = True
    window._preview_active = True
    window._sync_digital_slide_mode_ui()
    QApplication.processEvents()
    assert not window._inspector_dock.isVisible()
    assert window._capture_task_bar.isVisible()
    for button in (window._digital_slide_start_button, window._digital_slide_stop_button):
        assert button.isVisible()
        assert window.rect().contains(button.mapTo(window, button.rect().bottomRight()))
    window._sync_workspace_mode()
    assert window._capture_task_bar.isVisible()


def test_area_tool_changes_keep_subtraction_for_the_same_object(window):
    document = mount(window, 1)
    area = Measurement(
        id=new_id("measurement"), image_id=document.id, fiber_group_id=document.active_group_id,
        measurement_kind="area", mode="polygon_area",
        polygon_px=[Point(20, 20), Point(140, 20), Point(140, 130), Point(20, 130)],
    )
    document.add_measurement(area)
    window._update_ui_for_current_document()
    window.set_tool_mode("polygon_area")
    window._focus_current_canvas()
    QTest.keyClick(window.current_canvas(), Qt.Key.Key_T)
    assert window.current_canvas().current_area_edit_operation_mode() == AreaEditOperationMode.SUBTRACT
    window._measurement_context_bar.areaButtons[1].click()
    assert window._tool_mode == "freehand_area"
    assert window.current_canvas().current_area_edit_operation_mode() == AreaEditOperationMode.SUBTRACT
    assert window._area_operation_button.text() == "剔除(T)"
    QApplication.processEvents()
    assert window._area_operation_button.isVisible()
    assert window.rect().contains(window._area_operation_button.mapTo(window, window._area_operation_button.rect().bottomRight()))
    window._measurement_context_bar.areaButtons[0].click()
    assert window.current_canvas().current_area_edit_operation_mode() == AreaEditOperationMode.SUBTRACT


def test_magic_subtraction_keeps_shape_keys_and_adds_category_alternative(window):
    document = mount(window, 1)
    window.set_tool_mode("magic_segment")
    canvas = window.current_canvas()
    canvas._magic_segment.active_stage = MagicSegmentOperationMode.SUBTRACT
    window._update_magic_segment_controls()
    window._focus_current_canvas()
    QTest.keyClick(canvas, Qt.Key.Key_2)
    assert canvas.current_magic_subtract_input_mode() == MagicSegmentSubtractInputMode.POLYGON
    original_shape = canvas.current_magic_subtract_input_mode()
    QTest.keyClick(canvas, Qt.Key.Key_2, Qt.KeyboardModifier.AltModifier)
    assert document.active_group_id == document.get_group_by_number(2).id
    assert canvas.current_magic_subtract_input_mode() == original_shape
    window._cycle_magic_subtract_shape()
    assert canvas.current_magic_subtract_input_mode() == MagicSegmentSubtractInputMode.FREEHAND


def test_review_and_focus_restore_layout_and_keep_record_selection(window):
    document = mount(window, 1)
    window.resize(1280, 720)
    QApplication.processEvents()
    selected = document.measurements[0].id
    window._records_controller.select_measurement_id(selected)
    previous = [dock.isVisible() for dock in (window._project_dock, window._inspector_dock, window._results_dock)]
    preferences = window._app_settings.workspace_layout.to_dict()
    for mode in ("review", "focus"):
        window._set_workbench_presentation(mode)
        QApplication.processEvents()
        assert not window._project_dock.isVisible()
        assert not window._inspector_dock.isVisible()
        assert window._results_dock.isVisible() == (mode == "review")
        assert window._records_controller.selected_measurement_ids() == [selected]
        assert window._measurement_context_bar.isVisible()
        window._set_workbench_presentation("standard")
        QApplication.processEvents()
        assert [dock.isVisible() for dock in (window._project_dock, window._inspector_dock, window._results_dock)] == previous
        assert window._app_settings.workspace_layout.to_dict() == preferences


def test_project_page_keeps_images_and_categories_together(window):
    mount(window, 1)
    window.resize(1600, 900)
    QApplication.processEvents()
    assert window.image_list.isVisible()
    assert window.group_list.isVisible()
    window._project_navigation_tabs.setCurrentIndex(1)
    assert window._geometry_manager_tabs.isVisible()


def test_manual_panel_change_leaves_presentation_and_last_close_restores_it(window):
    mount(window, 1)
    window.resize(1280, 720)
    QApplication.processEvents()
    window._set_workbench_presentation("focus")
    window.toggle_project_panel_action.trigger()
    assert window._workbench_presentation_mode == "standard"
    assert window._project_dock.isVisible()
    assert not window.focus_workspace_action.isChecked()
    window._set_workbench_presentation("review")
    window.toggle_results_panel_action.trigger()
    assert window._workbench_presentation_mode == "standard"
    assert not window._results_dock.isVisible()
    preferences = window._app_settings.workspace_layout.to_dict()
    window._set_workbench_presentation("focus")
    with patch.object(window, "_confirm_close_documents", return_value=True):
        window.close_current_document()
    assert window._center_stack.currentWidget() is window._welcome_panel
    assert window._workbench_presentation_state is None
    assert window._app_settings.workspace_layout.to_dict() == preferences


def test_typing_does_not_change_snap_or_category(window):
    document = mount(window, 1)
    window.set_tool_mode("manual")
    original_group = document.active_group_id
    search = window._inspector_records_pane.search_edit
    assert isinstance(search, QLineEdit)
    search.setFocus()
    QTest.keyClicks(search, "b2t")
    QTest.keyClick(search, Qt.Key.Key_2, Qt.KeyboardModifier.AltModifier)
    assert search.text().startswith("b2t")
    assert window._tool_mode == "manual"
    assert not window.toggle_edge_snap_action.isChecked()
    assert document.active_group_id == original_group
    assert search.hasFocus()
    window._measurement_context_bar.snapButton.click()
    assert window._tool_mode == "snap"
    assert window.current_canvas().hasFocus()


def test_command_search_finds_alias_and_runs_existing_action_after_closing(window):
    mount(window, 1)
    dialog = CommandSearchDialog(window._command_search_entries(), window)
    dialog.search.setText("原始记录")
    assert dialog.results.count() == 1
    assert dialog.results.item(0).data(Qt.ItemDataRole.UserRole) is window.export_template_action
    assert dialog.run_button.isEnabled()
    window.export_template_action.setEnabled(False)
    dialog._choose()
    assert dialog.chosen_action is None
    window.export_template_action.setEnabled(True)
    dialog._choose()
    assert dialog.chosen_action is window.export_template_action
    with patch("fdm.ui.main_window.CommandSearchDialog", return_value=dialog), \
         patch.object(dialog, "exec", return_value=dialog.DialogCode.Accepted), \
         patch.object(window, "export_results") as export:
        window._open_command_search()
        assert export.call_args.args[0].include_excel


def test_screen_label_density_changes_pixels_but_preserves_exports_and_values(window, tmp_path):
    document = mount(window, 1)
    canvas = window.current_canvas()
    measurement_values = [measurement.to_dict() for measurement in document.measurements]
    settings_before = window._app_settings.to_dict()
    screenshots = {}
    outputs = []
    for mode in ("all", "selected", "hidden", "all"):
        window._set_screen_label_mode(mode)
        canvas.set_selected_measurement(None)
        QApplication.processEvents()
        screenshots[mode] = canvas.grab().toImage()
        output = tmp_path / f"overlay-{len(outputs)}.png"
        window._render_overlay_image(
            document, output, include_measurements=True, include_scale=False,
            render_mode=ExportImageRenderMode.FULL_RESOLUTION,
        )
        outputs.append(output.read_bytes())
    assert screenshots["all"] != screenshots["hidden"]
    assert screenshots["selected"] == screenshots["hidden"]
    assert all(output == outputs[0] for output in outputs)
    assert [measurement.to_dict() for measurement in document.measurements] == measurement_values
    assert window._app_settings.to_dict() == settings_before
