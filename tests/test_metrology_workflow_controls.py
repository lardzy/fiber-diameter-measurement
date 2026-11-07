"""Exercise tool activation and caliper changes through the professional panel."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QRect

from fdm.geometry import Line, Point
from fdm.metrology import MeasurementDefinition, MeasurementFeature
from fdm.models import ProjectState
from fdm.ui.professional_measurement import TOOL_GROUPS, TOOLS

from test_metrology_workspace import mounted_window, _begin, _click, _mouse, _wait_until


def activate(panel, key):
    index = panel.tool.findData(key)
    assert index >= 0
    panel.tool.setCurrentIndex(index)
    panel.tool.activated.emit(index)


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_themed_controls_activate_step_back_and_repeat(mounted_window, desktop_application, theme):
    window, document, workspace, _ = mounted_window()
    window._app_settings.theme_mode = theme
    window._apply_theme_mode()
    window.resize(1067, 660)
    panel = workspace.panel
    assert sorted(key for keys in TOOL_GROUPS.values() for key in keys) == sorted(TOOLS)
    activate(panel, "point_line_distance")
    desktop_application.processEvents()
    controller = workspace.controller()
    assert controller.key == "point_line_distance"
    assert panel.form.labelForField(panel.sources[1]).text() == "基准线"
    assert panel.follow_sources.isChecked()
    _click(controller.canvas, (35, 50))
    assert controller.slot == 1
    panel.back_button.click()
    assert controller.slot == 0 and not document.professional_measurements
    _click(controller.canvas, (35, 50))
    _click(controller.canvas, (80, 20))
    _click(controller.canvas, (80, 120))
    assert controller.results()[document.professional_measurements[-1].id].primary.value == pytest.approx(45)
    assert panel.start_button.text() == "再测一次"
    panel.start_button.click()
    assert controller.key == "point_line_distance"
    assert not controller.points and controller.slot == 0
    area = workspace.dock.widget()
    area.ensureWidgetVisible(panel.follow_sources)
    desktop_application.processEvents()
    for control in (panel.follow_sources, panel.sources[0], panel.sources[1]):
        bounds = QRect(control.mapTo(area.viewport(), QPoint()), control.size())
        assert area.viewport().rect().contains(bounds), (bounds, area.viewport().rect())
    assert not window.grab().isNull()


def test_caliper_parameters_reuse_roi_and_invalidate_superseded_preview(mounted_window, desktop_application):
    _, document, workspace, _ = mounted_window()
    panel = workspace.panel
    activate(panel, "caliper_line")
    controller = workspace.controller()
    _click(controller.canvas, (30, 90))
    panel.width.setValue(15)
    assert len(controller.points) == 1
    _click(controller.canvas, (220, 90))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    old_job, old_result = workspace._preview
    panel.width.setValue(18)
    panel.count.setValue(24)
    assert not panel.confirm.isEnabled() and workspace._preview is None
    workspace._fit_finished(old_job, old_result)
    workspace.confirm_fit()
    assert not document.professional_measurements
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    job, _ = workspace._preview
    assert job["points"] == old_job["points"]
    assert job["options"].search_half_width_px == 18 and job["options"].sample_count == 24
    panel.confirm.click()
    assert len(document.professional_measurements) == 1
    assert document.professional_features[0].parameters["caliper"]["sample_count"] == 24
    panel.count.setValue(25)
    assert not workspace._refit_timer.isActive()


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_themed_controls_open_ordinary_label_in_narrow_inspector(mounted_window, desktop_application, theme):
    window, document, workspace, _ = mounted_window()
    window._app_settings.theme_mode = theme
    window._apply_theme_mode()
    window.resize(1067, 660)
    window.set_tool_mode("manual")
    for start, end in [((35, 66), (100, 74)), ((100, 74), (165, 82))]:
        window._on_canvas_line_committed(document.id, "manual", Line(Point(*start), Point(*end)))
    selected = document.measurements[-1]
    workspace.legacy_properties(document.id, selected.id)
    desktop_application.processEvents()
    editor = window._object_inspector.annotation_editor
    assert editor.isVisible() and editor.default_button.text() == "设为普通直径线默认"
    viewport = window._inspector_scroll.viewport()
    field = editor._fields["style"]
    assert viewport.rect().contains(QRect(field.mapTo(viewport, QPoint()), field.size()))
    assert selected.annotation_layout.label_position == "near"


def test_refit_keeps_a_datum_referenced_by_another_professional_result(mounted_window, desktop_application):
    window, document, workspace, _ = mounted_window()
    _begin(workspace, "caliper_line")
    controller = workspace.controller()
    _click(controller.canvas, (30, 90))
    _click(controller.canvas, (220, 90))
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    workspace.confirm_fit()
    source, = document.professional_measurements
    datum_id, = source.feature_ids
    reference = MeasurementFeature("linked", document.id, "line", professional_source_id=datum_id)
    child = MeasurementDefinition("other", document.id, "distance", (reference.id,))
    document.apply_professional_state([*document.professional_features, reference],
                                      [source, child], selected_id=source.id)
    assert "professional-source-bindings/v1" in ProjectState("linked", [document]).effective_required_features()
    window.set_tool_mode("manual")
    workspace.select(source.id, navigate=True)
    workspace.refit_selected()
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    assert controller.canvas._tool_mode == "professional"
    pos = controller.canvas.rect().center().toPointF()
    controller.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, pos))
    assert controller.canvas._drawing_line is None
    controller.canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, pos))
    assert not document.measurements
    workspace.confirm_fit()
    assert document.get_professional_measurement(source.id).feature_ids == (datum_id,)
    assert controller.results()[child.id].valid
