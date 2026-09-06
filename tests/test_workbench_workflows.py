from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QItemSelectionModel, QPoint, Qt
from PySide6.QtGui import QFont, QImage, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QStyle, QStyleOptionToolButton, QToolButton

from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, new_id
from fdm.settings import AppSettings, AppThemeMode, RawRecordTemplate
from fdm.services.export_service import ExportImageRenderMode
from fdm.ui.image_loader import ImageLoadRequest
from fdm.ui.main_window import MainWindow
from fdm.ui.canvas import AreaEditOperationMode, MagicSegmentOperationMode, MagicSegmentSubtractInputMode
from fdm.ui.workbench_controls import CommandSearchDialog
from fdm.ui import icons, theme


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


def settle():
    for _ in range(4):
        QApplication.processEvents()


@pytest.mark.parametrize("width", [1093, 1512])
def test_legacy_toolbar_rows_recover_labels_through_project_close(window, width):
    window.resize(width, 800)
    window.removeToolBarBreak(window._context_toolbar)
    legacy_state = window.saveState(2)
    assert window.restoreState(legacy_state, 2)
    documents = [mount(window, i) for i in range(12)]
    window.set_tool_mode("count")
    window._records_controller.select_measurement_id(documents[-1].measurements[0].id)
    settle()
    for _ in range(2):
        tools, context = window._measure_toolbar, window._context_toolbar
        assert window.toolBarBreak(tools)
        assert window.toolBarBreak(context)
        assert context.y() >= tools.geometry().bottom()
        assert tools.width() == window.width()
        assert not window._measurement_tool_strip._mode_buttons["count"].isCompactMode()
        with patch.object(window, "_confirm_close_documents", return_value=True):
            window.close_all_documents()
        settle()
    assert window.current_document() is None


def test_minimum_project_width_contains_lists_headers_and_actions(window):
    document = mount(window, 1)
    document.create_group(label="用于验证窄栏的超长纤维类别名称", color="#446688")
    window._populate_group_list(document)
    window.resize(1512, 863)
    settle()
    window._project_dock.show()
    window.resizeDocks([window._project_dock], [220], Qt.Orientation.Horizontal)
    settle()
    viewport = window._left_standard_splitter.viewport()
    content = window._left_standard_splitter.widget()
    assert window._project_dock.width() <= 222
    assert content.width() == viewport.width()
    for control in (window.image_list, window.group_list, *window._group_header_labels,
                    window._add_group_button, window._rename_group_button, window.delete_group_button):
        assert control.isVisible()
        assert control.mapTo(viewport, QPoint(0, 0)).x() >= 0
        assert control.mapTo(viewport, control.rect().bottomRight()).x() < viewport.width()
    for index in range(window.group_list.count()):
        item = window.group_list.itemWidget(window.group_list.item(index))
        assert item.width() <= window.group_list.viewport().width()
        assert item.mapTo(window.group_list.viewport(), item.rect().bottomRight()).x() < window.group_list.viewport().width()


def test_selection_summary_reserves_height_for_empty_single_and_multiple(window):
    document = mount(window, 1)
    document.path = "/virtual/" + "很长的图片文件名" * 20 + ".png"
    first = document.measurements[0]
    second = Measurement(id=new_id("measurement"), image_id=document.id, fiber_group_id=first.fiber_group_id,
                         mode="manual", line_px=Line(Point(20, 50), Point(100, 50)), confidence=1, status="manual")
    document.add_measurement(second)
    window._populate_measurement_table(document)
    controller = window._records_controller
    controller.select_measurement_id(None)
    settle()
    panel = window._current_measurement_summary
    geometry = (panel.height(), window._statistics_section.y(), window._records_section.y())
    for selection in ([first.id], [first.id, second.id], [], [second.id]):
        controller.select_measurement_id(selection[0] if selection else None)
        if len(selection) > 1:
            index = controller.proxy_model.index(1, 0)
            controller.selection_model.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        settle()
        assert (panel.height(), window._statistics_section.y(), window._records_section.y()) == geometry
        assert panel.groupCombo.isVisible()
        assert panel.groupCombo.isEnabled() == (len(selection) == 1)
    assert Path(document.path).name in panel.sourceLabel.toolTip()


def test_current_object_properties_button_toggles_and_tracks_section(window):
    mount(window, 1)
    button = window._current_measurement_summary.editButton
    section = window._object_properties_section
    for expanded in (True, False, True):
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        settle()
        assert section.isExpanded() == expanded
        assert section.isVisible() == expanded
        assert button.isChecked() == expanded
    section.toggleButton.click()
    assert not button.isChecked()


def test_inspector_count_has_one_badge_and_keeps_filter_totals(window):
    mount(window, 1)
    pane = window._inspector_records_pane
    badge = window._records_section.summaryLabel
    settle()
    assert not pane.count_label.isVisible()
    assert badge.text() == "1 条"
    assert badge.width() < 90
    pane.search_edit.setText("没有匹配的测量")
    settle()
    assert badge.text() == "0 / 1 条"
    assert pane.table.model().rowCount() == 0
    pane.search_edit.clear()
    assert pane.table.model().rowCount() == 1


def test_context_category_stays_adjacent_at_both_densities(window):
    mount(window, 1)
    window.set_tool_mode("polygon_area")
    window._set_workbench_presentation("focus")
    for width in (1512, 800):
        window.resize(width, 800)
        settle()
        bar = window._measurement_context_bar
        label = bar.findChild(QLabel, "measurementCategoryLabel")
        label_right = label.mapTo(bar, label.rect().topRight()).x()
        combo_left = bar.groups.mapTo(bar, QPoint(0, 0)).x()
        assert 4 <= combo_left - label_right <= 12
        for control in (bar.groups, bar.documents, bar.calibrationButton, window._area_operation_button):
            assert bar.rect().contains(control.mapTo(bar, control.rect().bottomRight()))


def test_object_snap_split_button_keeps_text_clear_of_menu(window):
    button = window._object_snap_status_button
    option = QStyleOptionToolButton()
    button.initStyleOption(option)
    main = button.style().subControlRect(QStyle.ComplexControl.CC_ToolButton, option,
                                         QStyle.SubControl.SC_ToolButton, button)
    arrow = button.style().subControlRect(QStyle.ComplexControl.CC_ToolButton, option,
                                          QStyle.SubControl.SC_ToolButtonMenu, button)
    assert arrow.width() >= 20
    assert main.width() - button.fontMetrics().horizontalAdvance(button.text()) >= 12
    before = button.isChecked()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=main.center())
    assert button.isChecked() != before
    assert len(button.menu().actions()) >= 7


@pytest.mark.parametrize("fallback", [False, True])
def test_count_and_auxiliary_point_icons_are_distinct_at_small_size(desktop_application, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(icons, "qta", None)
    count = icons.themed_icon("count", color="#556677", size=16).pixmap(16, 16).toImage()
    point = icons.themed_icon("point", color="#556677", size=16).pixmap(16, 16).toImage()
    assert not count.isNull()
    assert count != point
    assert sum(count.pixelColor(x, y).alpha() > 0 for y in range(16) for x in range(16)) > 30


def test_system_theme_uses_stable_controls_and_tracks_appearance(window, monkeypatch):
    app = QApplication.instance()
    window.resize(1291, 832)
    original = window._app_settings.theme_mode
    color_mode = AppThemeMode.DARK
    monkeypatch.setattr(theme, "_system_color_mode", lambda _app: color_mode)
    try:
        theme.apply_application_theme(app, AppThemeMode.SYSTEM)
        settle()
        geometry = (window._measure_toolbar.height(), window._context_toolbar.height())
        icon_sizes = (window._file_toolbar.iconSize(), window._manual_tool_button.iconSize())
        for color_mode, scheme, palette in (
            (AppThemeMode.LIGHT, Qt.ColorScheme.Light, theme.build_light_palette()),
            (AppThemeMode.DARK, Qt.ColorScheme.Dark, theme.build_dark_palette()),
        ):
            app.styleHints().colorSchemeChanged.emit(scheme)
            settle()
            assert app.property("fdmAppliedThemeMode") == AppThemeMode.SYSTEM
            assert app.property("fdmBaseWidgetStyle").casefold() == "fusion"
            assert app.palette().color(QPalette.ColorRole.Window) == palette.color(QPalette.ColorRole.Window)
            assert (window._measure_toolbar.height(), window._context_toolbar.height()) == geometry
            assert (window._file_toolbar.iconSize(), window._manual_tool_button.iconSize()) == icon_sizes
            search = window._file_toolbar.widgetForAction(window.command_search_action)
            assert search.isVisible()
            assert window.rect().contains(search.mapTo(window, search.rect().bottomRight()))
        theme.apply_application_theme(app, AppThemeMode.DARK)
        color_mode = AppThemeMode.LIGHT
        app.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Light)
        settle()
        assert app.palette().color(QPalette.ColorRole.Window) == theme.build_dark_palette().color(QPalette.ColorRole.Window)
        with patch.object(app, "setPalette") as apply_palette:
            theme.apply_application_theme(app, AppThemeMode.DARK)
        apply_palette.assert_not_called()
    finally:
        theme.apply_application_theme(app, original)


def test_theme_switch_preserves_widget_class_fonts(desktop_application):
    app = desktop_application
    theme.apply_application_theme(app, AppThemeMode.DARK)
    original = QFont(app.font("QToolButton"))
    compact_font = QFont(original)
    compact_font.setPointSize(9)
    app.setFont(compact_font, "QToolButton")
    button = QToolButton()
    button.setText("测量工具")
    button.ensurePolished()
    expected_size = button.sizeHint()
    try:
        for mode in (AppThemeMode.LIGHT, AppThemeMode.DARK):
            theme.apply_application_theme(app, mode)
            settle()
            assert button.font().pointSize() == 9
            assert button.sizeHint() == expected_size
    finally:
        button.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.setFont(original, "QToolButton")


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
