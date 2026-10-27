"""A fitting preview cannot overwrite changed semantics or lose its units."""
from __future__ import annotations

from dataclasses import replace
from threading import Event

import pytest

from fdm.geometry import Point
from fdm.models import Calibration
from fdm.ui.professional_measurement import refit_target_snapshot

from test_metrology_workspace import mounted_window, _begin, _click, _distance, _wait_until


def _preview(workspace, app):
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    _wait_until(app, lambda: workspace._preview is not None)
    assert workspace.panel.confirm.isEnabled(), workspace.panel.prompt.text()


def _saved_caliper(workspace, app):
    _preview(workspace, app)
    workspace.confirm_fit()
    return workspace.controller().canvas._document.professional_measurements[-1]


def test_unit_change_after_preview_discards_result_even_when_axis_scales_match(mounted_window, desktop_application):
    _window, document, workspace, _plane = mounted_window()
    document.calibration = Calibration("manual", 2, "um", "original")
    _preview(workspace, desktop_application)
    workspace.change(document, "变更标定单位", lambda: setattr(document, "calibration", replace(document.calibration, unit="nm")))
    workspace.confirm_fit()
    assert not document.professional_measurements
    assert workspace._preview is None and not workspace.panel.confirm.isEnabled()
    assert "来源已改变" in workspace.panel.prompt.text()


def test_unit_change_while_worker_runs_cannot_install_preview(mounted_window, monkeypatch, desktop_application):
    from fdm.services import metrology_fitting
    _window, document, workspace, _plane = mounted_window()
    document.calibration = Calibration("manual", 2, "um", "original")
    entered, release = Event(), Event()
    original = metrology_fitting.run_line_caliper

    def delayed(*args, **kwargs):
        entered.set()
        if not release.wait(2):
            raise RuntimeError("test worker gate timed out")
        return original(*args, **kwargs)

    monkeypatch.setattr(metrology_fitting, "run_line_caliper", delayed)
    _begin(workspace, "caliper_line")
    _click(workspace.controller().canvas, (30, 90))
    _click(workspace.controller().canvas, (220, 90))
    try:
        assert entered.wait(1)
        workspace.change(document, "变更标定单位", lambda: setattr(document, "calibration", replace(document.calibration, unit="nm")))
    finally:
        release.set()
    _wait_until(desktop_application, lambda: workspace._job is None)
    assert workspace._preview is None and not document.professional_measurements
    assert "标定已改变" in workspace.panel.prompt.text()


@pytest.mark.parametrize("calibration", [None, Calibration("manual", 2, "um", "双轴标定", 3)])
def test_confirm_preserves_original_full_calibration_as_detached_evidence(mounted_window, desktop_application, calibration):
    _window, document, workspace, _plane = mounted_window()
    document.calibration = calibration
    expected = calibration.to_dict() if calibration else None
    definition = _saved_caliper(workspace, desktop_application)
    feature = document.get_professional_feature(definition.feature_ids[0])
    assert "calibration_at_creation" in feature.parameters
    assert feature.parameters["calibration_at_creation"] == expected
    if document.calibration:
        document.calibration.unit = "nm"
        document.calibration.source_label = "new source"
        assert feature.parameters["calibration_at_creation"] == expected


def test_refit_deleted_target_discards_preview_without_success_message(mounted_window, desktop_application):
    _window, document, workspace, _plane = mounted_window()
    _saved_caliper(workspace, desktop_application)
    workspace.refit_selected()
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    workspace.delete_selected()
    workspace.confirm_fit()
    assert not document.professional_measurements
    assert workspace._preview is None and not workspace.panel.confirm.isEnabled()
    assert "丢弃过期拟合" in workspace.panel.prompt.text()
    assert "已保存" not in workspace.panel.prompt.text()


@pytest.mark.parametrize("change", ["parameters", "points", "quality", "missing_feature"])
def test_refit_semantically_changed_target_is_not_overwritten(mounted_window, desktop_application, change):
    _window, document, workspace, _plane = mounted_window()
    definition = _saved_caliper(workspace, desktop_application)
    feature = document.get_professional_feature(definition.feature_ids[0])
    workspace.refit_selected()
    _wait_until(desktop_application, lambda: workspace._preview is not None)

    def mutate():
        if change == "parameters":
            rows = [replace(d, parameters={"review_revision": 2}) if d.id == definition.id else d for d in document.professional_measurements]
            document.apply_professional_state(document.professional_features, rows, mark_dirty=False)
        elif change == "missing_feature":
            document.professional_features = []
            document.mark_professional_geometry_changed()
        else:
            replacement = replace(feature, points_px=tuple(Point(p.x, p.y + 1) for p in feature.points_px)) if change == "points" else replace(feature, quality={**feature.quality, "review_revision": 2})
            document.apply_professional_state([replacement], document.professional_measurements, mark_dirty=False)

    workspace.change(document, "变更拟合来源", mutate)
    target_after_edit = document.get_professional_measurement(definition.id)
    features_after_edit = tuple(document.professional_features)
    workspace.confirm_fit()
    assert document.get_professional_measurement(definition.id) == target_after_edit
    assert tuple(document.professional_features) == features_after_edit
    assert workspace._preview is None and "丢弃过期拟合" in workspace.panel.prompt.text()


def test_refit_allows_layout_selection_and_name_edits_and_keeps_the_latest_layout(mounted_window, desktop_application):
    _window, document, workspace, _plane = mounted_window()
    definition = _saved_caliper(workspace, desktop_application)
    other = _distance(workspace, (30, 130), (100, 160))
    workspace.select(definition.id)
    workspace.refit_selected()
    _wait_until(desktop_application, lambda: workspace._preview is not None)
    moved = definition.annotation_layout.with_manual_offset(Point(120, -65))
    workspace.edit_layout(workspace.controller(), definition.id, moved)
    workspace.change(document, "修改名称", lambda: document.apply_professional_state(
        document.professional_features,
        [replace(d, name="已复核") if d.id == definition.id else d for d in document.professional_measurements],
        mark_dirty=False))
    workspace.select(other.id)
    workspace.confirm_fit()
    refreshed = document.get_professional_measurement(definition.id)
    assert len(document.professional_measurements) == 2
    assert refreshed.feature_ids != definition.feature_ids
    assert refreshed.annotation_layout == moved and refreshed.name == "已复核"
    assert document.get_professional_measurement(other.id) == other
    assert "已保存" in workspace.panel.prompt.text()


def test_refit_snapshot_tracks_every_referenced_feature_not_only_the_first(mounted_window, desktop_application):
    _window, document, workspace, _plane = mounted_window()
    first = _saved_caliper(workspace, desktop_application)
    second = _distance(workspace, (30, 130), (100, 160))
    combined = replace(first, feature_ids=first.feature_ids + second.feature_ids)
    document.apply_professional_state(document.professional_features, [combined, second], mark_dirty=False)
    before = refit_target_snapshot(document, first.id)
    changed_id = second.feature_ids[0]
    document.apply_professional_state(
        [replace(f, points_px=(Point(40, 130), Point(120, 160))) if f.id == changed_id else f for f in document.professional_features],
        document.professional_measurements, mark_dirty=False)
    assert refit_target_snapshot(document, first.id) != before
